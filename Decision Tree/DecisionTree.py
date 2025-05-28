# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

# Load the dataset from an Excel file
df = pd.read_excel("Iris.xlsx")

# Inspect the first few rows to ensure the data is loaded correctly
print("Initial Dataset:")
print(df.head())

# Preprocessing the dataset
df = df.drop(columns=['Id'])  # Drop the 'Id' column
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']  # Rename columns
df['species'] = df['species'].str.replace('Iris-', '')  # Remove 'Iris-' prefix from species names

# Convert the dataframe to a list of lists for easier manipulation
dataset = df.values.tolist()

# Display the first 5 rows after preprocessing
print("\nDataset after preprocessing (first 5 rows):")
print(df.head())
print("\n")

# Helper function to calculate the Gini Index
def gini_index(groups, classes):
    total_samples = sum([len(group) for group in groups])  # Total number of samples
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:  # Avoid division by zero
            continue
        score = 0.0
        class_counts = Counter([row[-1] for row in group])  # Count occurrences of each class in the group
        for class_val in classes:
            proportion = class_counts[class_val] / size  # Proportion of each class in the group
            score += proportion * proportion  # Sum of squared proportions
        gini += (1 - score) * (size / total_samples)  # Weighted Gini index for the group
    return gini

# Function to split the dataset based on an attribute and a threshold value
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:  # If the feature value is less than the threshold, go to the left group
            left.append(row)
        else:  # Otherwise, go to the right group
            right.append(row)
    return left, right

# Select the best split by checking all features and possible splits
def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))  # Get unique class labels (target values)
    best_index, best_value, best_score, best_groups = None, None, float('inf'), None  # Initialize variables
    for index in range(len(dataset[0]) - 1):  # Iterate through all features (exclude target column)
        for row in dataset:
            groups = test_split(index, row[index], dataset)  # Split dataset on this feature and value
            gini = gini_index(groups, class_values)  # Calculate Gini index for this split
            if gini < best_score:  # Check if this split is better (lower Gini index)
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

# Create a terminal node value (the most common output in a group)
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Build the decision tree recursively by splitting nodes until stopping criteria are met
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])  # Remove groups from node once they're split
    
    # Check if either left or right group is empty (no further splitting possible)
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    
    # Check if maximum depth is reached
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    
    # Process left child node if it meets minimum size criteria; otherwise make it a terminal node
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    
    # Process right child node if it meets minimum size criteria; otherwise make it a terminal node
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right)
        split(node['right'], max_depth, min_size, depth + 1)

# Build a decision tree from training data
def build_tree(train_data, max_depth, min_size):
    root = get_best_split(train_data)  # Get the root node by finding the best split point
    split(root, max_depth, min_size, depth=1)  # Recursively split nodes starting from root
    return root

# Make a prediction with a decision tree for one row of data
def predict(node, row):
    if row[node['index']] < node['value']:  # Check which side of the split the row belongs to (left or right)
        if isinstance(node['left'], dict):  # If left child is another decision node (not terminal), recurse further
            return predict(node['left'], row)
        else:
            return node['left']  # If terminal node is reached, return its value (class label)
    else:
        if isinstance(node['right'], dict):  # Same logic applies to right child nodes
            return predict(node['right'], row)
        else:
            return node['right']

# Make predictions for an entire dataset using a decision tree model
def predict_dataset(tree, test_data):
    predictions = [predict(tree, row) for row in test_data]
    return predictions

# Function to plot the decision tree structure using matplotlib.
def plot_tree(node, depth=0, x=0.5, y=1.0, ax=None):
    if ax is None: 
        fig, ax = plt.subplots(figsize=(12, 8))
    
    if isinstance(node, dict): 
        text = f"X{node['index']} < {node['value']:.2f}"
        ax.text(x, y, text,
                ha='center', va='center',
                bbox=dict(boxstyle='round', fc='lightblue'))
        
        # Left child plotting coordinates.
        ax.plot([x, x - (0.2 / (depth + 1))], [y - 0.05 * depth * depth + 0.05,
                                               y - (depth + 1) * 0.1], 'k-')
        
        plot_tree(node['left'], depth + 1,
                  x - (0.2 / (depth + 1)), y - (depth + 1) * 0.1,
                  ax=ax)

        # Right child plotting coordinates.
        ax.plot([x, x + (0.2 / (depth + 1))], [y - 0.05 * depth * depth + 0.05,
                                               y - (depth + 1) * 0.1], 'k-')

        plot_tree(node['right'], depth + 1,
                  x + (0.2 / (depth + 1)), y - (depth + 1) * 0.1,
                  ax=ax)

# Split dataset into training and testing sets (80% train / 20% test)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Set hyperparameters for decision tree model
max_depth = 5   # Maximum depth of the tree (stopping criterion)
min_size = 10   # Minimum number of samples required to keep splitting

# Build the decision tree using training data and specified hyperparameters
tree = build_tree(train_data, max_depth=max_depth, min_size=min_size)

# Predict on test data and calculate accuracy of predictions
predictions = predict_dataset(tree, test_data)

actual_labels = [row[-1] for row in test_data]  # Extract actual labels from test data

accuracy = sum([1 if pred == actual else 0 for pred, actual in zip(predictions, actual_labels)]) / len(actual_labels)

print(f"Accuracy: {accuracy * 100:.2f}%")   # Print accuracy percentage

# Plotting the decision tree structure.
plt.title('Decision Tree Visualization')
plot_tree(tree)
plt.show()
