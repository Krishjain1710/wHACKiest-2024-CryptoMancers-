import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('D:/Quant Maze 2.0/hashtest.csv')

# Check if the dataset contains necessary columns
if 'Hash' not in df.columns or 'Algorithm' not in df.columns:
    raise ValueError("Dataset must contain 'Hash' and 'Algorithm' columns.")

# Convert hex hash values to numeric arrays and pad to a fixed length
def hex_to_padded_array(hex_str, target_length=64):
    hex_str = hex_str.replace(" ", "")  # Remove spaces if any
    array = np.array([int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)], dtype=np.float32)
    if len(array) < target_length:
        array = np.pad(array, (0, target_length - len(array)), 'constant')
    return array

# Set the target length to 64 for SHA-512 (longest hash in the dataset)
target_length = 64
df['Hash'] = df['Hash'].apply(lambda x: hex_to_padded_array(x, target_length))

# Encode algorithm labels
label_encoder = LabelEncoder()
df['Algorithm'] = label_encoder.fit_transform(df['Algorithm'])

# Prepare data for training
X = np.stack(df['Hash'].values)
y = df['Algorithm'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a PyTorch Dataset
class HashDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset and dataloaders
train_dataset = HashDataset(X_train, y_train)
test_dataset = HashDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network model
class HashClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HashClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = target_length
num_classes = len(label_encoder.classes_)
model = HashClassifier(input_size=input_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Average loss over the epoch
    epoch_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Initialize dictionaries to track correct and total counts per algorithm
algorithm_correct = {algorithm: 0 for algorithm in range(num_classes)}
algorithm_total = {algorithm: 0 for algorithm in range(num_classes)}

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)
        
        # Update correct and total counts per algorithm
        for i, label in enumerate(target):
            algorithm_total[label.item()] += 1
            if predicted[i] == label:
                algorithm_correct[label.item()] += 1

# Calculate overall accuracy
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Calculate and print accuracy for each algorithm
algorithm_accuracy = {}
for algorithm in range(num_classes):
    if algorithm_total[algorithm] > 0:
        algorithm_accuracy[label_encoder.inverse_transform([algorithm])[0]] = \
            (algorithm_correct[algorithm] / algorithm_total[algorithm]) * 100

# Print algorithm-wise accuracy
for algorithm, acc in algorithm_accuracy.items():
    print(f"Accuracy for {algorithm}: {acc:.2f}%")

# Plotting the accuracy of each cryptographic algorithm
algorithms = list(algorithm_accuracy.keys())
accuracies = list(algorithm_accuracy.values())

plt.figure(figsize=(10, 6))
plt.barh(algorithms, accuracies, color='skyblue')
plt.xlabel('Accuracy (%)')
plt.title('Accuracy per Cryptographic Algorithm')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.show()

# Save the trained model (recommended method)
torch.save(model.state_dict(), 'hash_classifier_model.pth')

print("Model saved successfully!")
