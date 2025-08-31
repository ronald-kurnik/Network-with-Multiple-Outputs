import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import h5py
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # Import for the linear regression line


hdf5_test_filename = 'DigitsDataTest.h5'

history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'val_rmse': []
}

with h5py.File('DigitsDataTrain.h5', 'r') as f:
    # Access datasets by their names
    anglesTrain = f['/anglesTrain'][:]
    labelsTrain = f['/labelsTrain'][:]
    XTrain = f['/XTrain'][:]

anglesTrain = anglesTrain.astype(np.float32)
labelsTrain = labelsTrain.astype(np.float32)
XTrain = XTrain.astype(np.float32)
anglesTrain = anglesTrain.T
labelsTrain = labelsTrain.T
labelsTrain = labelsTrain - 1
print("Training data loaded successfully from HDF5 file.")
print(f"anglesTrain shape: {anglesTrain.shape}")
print(f"labelsTrain shape: {labelsTrain.shape}")
print(f"XTrain shape: {XTrain.shape}")


try:
    with h5py.File(hdf5_test_filename, 'r') as f:
        anglesTest = f['/anglesTest'][:]
        labelsTest = f['/labelsTest'][:]
        XTest = f['/XTest'][:]

    anglesTest = anglesTest.astype(np.float32)
    labelsTest = labelsTest.astype(np.float32)
    XTest = XTest.astype(np.float32)
    anglesTest = anglesTest.T
    labelsTest = labelsTest.T
    labelsTest = labelsTest - 1
    print("Test data loaded successfully from HDF5 file.")
    print(f"anglesTest shape: {anglesTest.shape}")
    print(f"labelsTest shape: {labelsTest.shape}")
    print(f"XTest shape: {XTest.shape}")
except KeyError as e:
    print(f"Error loading test data: {e}. Please ensure the correct file and dataset names are used.")
    # Exit or handle the error gracefully if the file is not found
except FileNotFoundError as e:
    print(f"Error: {hdf5_test_filename} not found. Please ensure the file exists in the same directory.")
    # Exit or handle the error gracefully


class DigitsDataset(Dataset):
    def __init__(self, images, labels, angles, transform=None):
       
        self.images = np.transpose(images, (0, 2, 3, 1)).copy() # From (N, C, H, W) to (N, H, W, C)

        self.labels = labels
        self.angles = angles
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx] # Now image should be (28, 28, 1)

        # Ensure label is a single scalar if it's (1,)
        label = self.labels[idx].item() if self.labels[idx].ndim > 0 else self.labels[idx]
        angle = self.angles[idx].item() if self.angles[idx].ndim > 0 else self.angles[idx]


        if self.transform:
            # transforms.ToTensor() will convert (H, W, C) -> (C, H, W) and to torch.float32
            image = self.transform(image)
        
        # Ensure label is a LongTensor and angle is a FloatTensor (and potentially shaped for targets)
        label_tensor = torch.tensor(label, dtype=torch.long) # For CrossEntropyLoss
        angle_tensor = torch.tensor(angle, dtype=torch.float32) # For MSELoss

        return image, label_tensor, angle_tensor

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = DigitsDataset(XTrain, labelsTrain, anglesTrain, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = DigitsDataset(XTest, labelsTest, anglesTest, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class MultiOutputNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MultiOutputNet, self).__init__()
        # Main branch

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2) # 28x28 -> 28x28 (stride=1, padding=(5-1)/2=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu_activation = nn.ReLU() # Renamed from relu1

        # First downsampling layer
        # 28x28 -> 14x14 (stride=2, kernel=3, padding=1)
        # Formula: floor((28 - 3 + 2*1)/2) + 1 = floor(27/2) + 1 = 13 + 1 = 14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # Fixed padding
        self.bn2 = nn.BatchNorm2d(32)

        # Second downsampling layer to reach 7x7
        # 14x14 -> 7x7 (stride=2, kernel=3, padding=1)
        # Formula: floor((14 - 3 + 2*1)/2) + 1 = floor(13/2) + 1 = 6 + 1 = 7
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1) # Added stride=2, fixed padding
        self.bn3 = nn.BatchNorm2d(32)
        # Note: If you want another non-strided conv after this, add it as conv4.
        # Your original conv3 was non-strided. For 7x7, this conv3 must be strided.

        # Skip connection
        # This skip connection also needs to perform two levels of downsampling (28x28 -> 7x7)
        # Stride of 2 means (28 -> 14), another stride of 2 means (14 -> 7).
        # One can either use a single conv with stride=4 (if kernel allows) or two sequential convs.
        # Given the original structure, two sequential skip convs align better.

        # First part of skip connection (28x28 -> 14x14)
        self.conv_skip1 = nn.Conv2d(16, 32, kernel_size=1, stride=2) # Matches conv2 downsampling
        self.bn_skip1 = nn.BatchNorm2d(32)

        # Second part of skip connection (14x14 -> 7x7)
        self.conv_skip2 = nn.Conv2d(32, 32, kernel_size=1, stride=2) # Matches conv3 downsampling
        self.bn_skip2 = nn.BatchNorm2d(32)


        # Calculate input features for fully connected layers
        # With two stride-2 conv layers, and starting from 28x28:
        # 28 -> (conv2 stride 2) -> 14 -> (conv3 stride 2) -> 7
        fc_input_features = 32 * 7 * 7 # This is now correct: 32 channels * 7 * 7 spatial = 1568

        # Classification branch
        self.fc_classification = nn.Linear(fc_input_features, num_classes)

        # Regression branch
        self.fc_regression = nn.Linear(fc_input_features, 1)

    def forward(self, x):
        # Main branch
        x_conv1_out = F.relu(self.bn1(self.conv1(x)))
        x_main = F.relu(self.bn2(self.conv2(x_conv1_out))) # Input is x_conv1_out (16 channels)
        x_main = F.relu(self.bn3(self.conv3(x_main))) # Input is x_main (32 channels)

        # Skip connection
        x_skip = F.relu(self.bn_skip1(self.conv_skip1(x_conv1_out))) # Input is x_conv1_out (16 channels)
        x_skip = F.relu(self.bn_skip2(self.conv_skip2(x_skip))) # Input is x_skip (32 channels)

        # Add skip connection to main branch output
        # Ensure dimensions match before adding
        x_combined = x_main + x_skip

        # Flatten for fully connected layers
        x_flattened = x_combined.view(x_combined.size(0), -1)

        # Classification branch
        classification_output = self.fc_classification(x_flattened)

        # Regression branch
        regression_output = self.fc_regression(x_flattened)

        return classification_output, regression_output
		
criterion_classification = nn.CrossEntropyLoss()
criterion_regression = nn.MSELoss()

def custom_loss_function(y_pred_classification, y_pred_regression, y_true_classification, y_true_regression, lambda_reg=0.1):
    loss_cls = criterion_classification(y_pred_classification, y_true_classification)
    loss_reg = criterion_regression(y_pred_regression, y_true_regression)
    return loss_cls + lambda_reg * loss_reg
	

# Instantiate the model
model = MultiOutputNet(num_classes=10)

# Define optimizer
learning_rate = 0.001  # Or any other value you want
optimizer = Adam(model.parameters(), lr=learning_rate)
# optimizer = Adam(model.parameters())

num_epochs = 30 # Adjust as needed
# lambda_regression = 0.1 # Weight for regression loss, as in MATLAB example
lambda_regression = 0.003 # Weight for regression loss, as in MATLAB example


# Early Stopping Parameters
patience = 5 # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# Move model to appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Training on: {device}")

for epoch in range(num_epochs):
    if early_stop:
        print("Early stopping triggered.")
        break

    # --- Training phase ---
    model.train() # Set model to training mode
    total_train_loss = 0
    for images, labels, angles in train_dataloader:
        images, labels, angles = images.to(device), labels.to(device), angles.to(device)
        optimizer.zero_grad()
        classification_output, regression_output = model(images)
        loss = custom_loss_function(classification_output, regression_output, labels, angles.unsqueeze(1), lambda_regression)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    history['train_loss'].append(avg_train_loss)
    
    # --- Validation phase ---
    model.eval() # Set model to evaluation mode
    total_val_loss = 0
    all_predictions_cls = []
    all_true_cls = []
    all_predictions_reg = []
    all_true_reg = []
    
    with torch.no_grad():
        for images, labels, angles in test_dataloader: # Using test_dataloader as validation
            images, labels, angles = images.to(device), labels.to(device), angles.to(device)
            classification_output, regression_output = model(images)
            loss = custom_loss_function(classification_output, regression_output, labels, angles.unsqueeze(1), lambda_regression)
            total_val_loss += loss.item()
            
            # For classification accuracy
            _, predicted_labels = torch.max(classification_output, 1)
            all_predictions_cls.extend(predicted_labels.cpu().numpy())
            all_true_cls.extend(labels.cpu().numpy())
            
            # For regression RMSE
            all_predictions_reg.extend(regression_output.cpu().numpy().flatten())
            all_true_reg.extend(angles.cpu().numpy().flatten())

    avg_val_loss = total_val_loss / len(test_dataloader)
    accuracy = accuracy_score(all_true_cls, all_predictions_cls)
    rmse = math.sqrt(mean_squared_error(all_true_reg, all_predictions_reg))
    
    history['val_loss'].append(avg_val_loss)
    history['val_accuracy'].append(accuracy)
    history['val_rmse'].append(rmse)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.4f}, Val RMSE: {rmse:.4f}")

    # --- Early Stopping Check ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            early_stop = True
            
# Calculate metrics from the final state of the model
final_accuracy = accuracy_score(all_true_cls, all_predictions_cls)
final_rmse = math.sqrt(mean_squared_error(all_true_reg, all_predictions_reg))

print(f"Classification Accuracy: {final_accuracy:.4f}")
print(f"Regression RMSE: {final_rmse:.4f}")
	
	
# --- Plotting the training and validation metrics ---

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.show()

# Plot Classification Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history['val_accuracy'], label='Validation Accuracy', color='green')
plt.title('Validation Classification Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.show()

# Plot Regression RMSE
plt.figure(figsize=(10, 5))
plt.plot(history['val_rmse'], label='Validation RMSE', color='red')
plt.title('Validation Regression RMSE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.savefig('rmse_plot.png')
plt.show()	
		
		
# --- Calculate per-class accuracy and print a confusion matrix ---

# Calculate the confusion matrix
cm = confusion_matrix(all_true_cls, all_predictions_cls)
print("\nConfusion Matrix (Rows: True Labels, Columns: Predicted Labels):")
print(cm)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10), cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Calculate and print per-class accuracy
class_accuracies = cm.diagonal() / cm.sum(axis=1)
print("\nAccuracy per digit:")
for i in range(10):
    print(f"Digit {i}: {class_accuracies[i]:.4f}")

# Sort and print digits by accuracy
sorted_indices = np.argsort(class_accuracies)[::-1]
print("\nDigits sorted by accuracy (most accurate first):")
for i in sorted_indices:
    print(f"Digit {i}: {class_accuracies[i]:.4f}")

# --- Scatter plot of predicted vs. actual angles with a regression line ---
plt.figure(figsize=(10, 10))
plt.scatter(all_true_reg, all_predictions_reg, alpha=0.5, label='Predicted vs. Actual Angles')

# Add a linear regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(all_true_reg, all_predictions_reg)
line = slope * np.array(all_true_reg) + intercept
plt.plot(all_true_reg, line, color='red', label=f'Regression Line (R² = {r_value**2:.2f})')
plt.plot(all_true_reg, all_true_reg, color='k', linestyle='--', label='Ideal Fit') # Ideal 45-degree line

plt.title('Predicted Angles vs. Actual Angles')
plt.xlabel('Actual Angle (Degrees)')
plt.ylabel('Predicted Angle (Degrees)')
plt.legend()
plt.grid(True)
plt.axis('equal') # Set equal scaling on both axes
plt.savefig('angle_regression_plot.png')
plt.show()