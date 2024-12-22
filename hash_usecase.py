import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define the HashClassifier model
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

# Load the trained model
model = HashClassifier(input_size=64, num_classes=5)
model.load_state_dict(torch.load('hash_classifier_model.pth'))
model.eval()

# LabelEncoder used during training to decode the predictions back to algorithm names
label_encoder = LabelEncoder()
label_encoder.fit(['SHA-256', 'SHA-512', 'MD5', 'SHA-1', 'Blake2'])

# Function to process a hash and predict the cryptographic algorithm
def predict_algorithm(hex_hash):
    def hex_to_padded_array(hex_str, target_length=64):
        hex_str = hex_str.replace(" ", "")
        array = np.array([int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)], dtype=np.float32)
        if len(array) < target_length:
            array = np.pad(array, (0, target_length - len(array)), 'constant')
        return array

    target_length = 64
    hash_array = hex_to_padded_array(hex_hash, target_length)
    input_tensor = torch.tensor(hash_array, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = output.max(1)
        predicted_algorithm = label_encoder.inverse_transform([predicted.item()])[0]
        return predicted_algorithm

# Example usage
test_hash = "6f1ed002ab5595859014ebf0951522d9"  # Example MD5 hash
predicted_algorithm = predict_algorithm(test_hash)

print(f"The predicted algorithm for the given hash is: {predicted_algorithm}")
