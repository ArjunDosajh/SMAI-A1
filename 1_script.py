import numpy as np
import argparse
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import heapq

# Read the name of input file from command line
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, help='Path to input file')

args = parser.parse_args()
input_file = args.input_file

if input_file == None or input_file == "":
    print("Please provide path to input file")
    exit(0)

# if input file is invalid, print error
try:
    with open(input_file, 'rb') as f:
        pass
except FileNotFoundError:
    print("Invalid path to input file")
    exit(0)

# Load the dataset
dataset = np.load(input_file, allow_pickle=True)

data_labels = dataset[:][:, 3].copy()
unique_data_labels = np.array(list(set(data_labels)))

data_label_to_idx = {label : idx + 1 for idx, label in enumerate(unique_data_labels)}
idx_to_data_label = {idx + 1 : label for idx, label in enumerate(unique_data_labels)}

X_resnet = dataset[:][:, 1].copy()
X_vit = dataset[:][:, 2].copy()
X_resnet = [arr.squeeze() for arr in X_resnet]
X_vit = [arr.squeeze() for arr in X_vit]

X_resnet = np.array(X_resnet)
X_vit = np.array(X_vit)
y = np.array([data_label_to_idx[label] for label in data_labels])

class KNN:
    def __init__(self, k=3, distance_metric='euclidean', encoder_type='classification'):
        self.k = k
        self.distance_metric = distance_metric
        self.encoder_type = encoder_type

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'cosine':
            return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all samples in the training set
        distances = [self.distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]  
        # Return the most common class label among k nearest neighbors
        if self.encoder_type == 'classification':
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        else:  # Regression
            return np.mean(k_nearest_labels)

    def evaluate(self, X_val, y_val):
        y_pred = self.predict(X_val)
        
        f1 = f1_score(y_val, y_pred, average='macro')
        acc = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_val, y_pred, average='macro', zero_division=1)
                
        return {"F1 Score": f1, "Accuracy": acc, "Precision": precision, "Recall": recall}
    
X_resnet_train, X_resnet_val, y_resnet_train, y_resnet_val = train_test_split(X_resnet, y, test_size=0.2, random_state=42)
X_vit_train, X_vit_val, y_vit_train, y_vit_val = train_test_split(X_vit, y, test_size=0.2, random_state=42)

print("Using ResNet embeddings with euclidean distance metric and k = 4")

# Initialize and fit KNN classifier
knn = KNN(k=4, distance_metric='euclidean', encoder_type='classification')
knn.fit(X_resnet_train, y_resnet_train)

# Predict and evaluate on validation set
result = knn.evaluate(X_resnet_val, y_resnet_val)
f1 = result['F1 Score']
acc = result['Accuracy']
precision = result['Precision']
recall = result['Recall']

# print the above 4 values in a nice table
print("| Metric     | Score       |")
print("|------------|-------------|")
print(f"| Accuracy   | {acc:0.3}   |")
print(f"| F1 Score   | {f1:0.3}   |")
print(f"| Precision  | {precision:0.3}   |")
print(f"| Recall     | {recall:0.3}   |")
