import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# Load the dataset
file_path = r"D:/Quant Maze 2.0/ciphertext_dataset_training.csv"
data = pd.read_csv(file_path)

# Ensure 'Ciphertext' column exists and remove spaces in the ciphertext
data['Ciphertext'] = data['Ciphertext'].str.replace(' ', '')  # Remove spaces

# Label encoding for the algorithm column (target)
label_encoder = LabelEncoder()
data['Algorithm'] = label_encoder.fit_transform(data['Algorithm'])

# Split data into features (X) and target (y)
X = data['Ciphertext']  # Features
y = data['Algorithm']   # Target

# Feature transformation: Convert Ciphertext to numerical values
print("Converting ciphertext to numerical arrays...")
X_transformed = []
for ciphertext in tqdm(X, desc="Transforming Ciphertext"):
    byte_array = [int(ciphertext[i:i + 2], 16) for i in range(0, len(ciphertext), 2)]
    X_transformed.append(byte_array)

# Custom feature for Rail Fence: Calculate frequency transitions
def compute_frequency_transitions(ciphertext):
    transitions = sum(1 for i in range(1, len(ciphertext)) if ciphertext[i] != ciphertext[i - 1])
    return transitions

data['FrequencyTransitions'] = data['Ciphertext'].apply(lambda x: compute_frequency_transitions(x))

# Padding to make ciphertexts of equal length
print("Padding ciphertexts...")
X_padded = pad_sequences(X_transformed, maxlen=128, padding='post', dtype='int32')

# Append the custom feature
X_padded = np.hstack([X_padded, data['FrequencyTransitions'].values.reshape(-1, 1)])

# Split the data into train and test sets (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.15, random_state=42)

# Calculate class weights to handle imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200, max_depth=30, class_weight=class_weights_dict, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model and label encoder as pickle files
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Model evaluation
print("Model evaluation results:")
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
print(classification_rep)

# Calculate and display overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}")

# Extract precision, recall, and F1-score for each algorithm (class)
precision = []
recall = []
f1_score = []

for class_label in range(len(label_encoder.classes_)):
    precision_value = classification_rep[label_encoder.classes_[class_label]]['precision']
    recall_value = classification_rep[label_encoder.classes_[class_label]]['recall']
    f1_value = classification_rep[label_encoder.classes_[class_label]]['f1-score']
    
    precision.append(precision_value)
    recall.append(recall_value)
    f1_score.append(f1_value)

# Plotting the evaluation metrics for each algorithm
x = label_encoder.classes_

fig, ax = plt.subplots(figsize=(10, 6))

# Bar plot for precision, recall, and F1-score
width = 0.2  # Width of the bars
x_pos = np.arange(len(x))

# Plot the bars for precision, recall, and F1-score
ax.bar(x_pos - width, precision, width, label='Precision', color='skyblue')
ax.bar(x_pos, recall, width, label='Recall', color='lightgreen')
ax.bar(x_pos + width, f1_score, width, label='F1-score', color='salmon')

# Labeling
ax.set_xlabel('Algorithm')
ax.set_ylabel('Scores')
ax.set_title('Model Evaluation: Precision, Recall, F1-Score by Algorithm')
ax.set_xticks(x_pos)
ax.set_xticklabels(x, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()

# Verify saved files
print(f"Files in directory: {os.listdir()}")
