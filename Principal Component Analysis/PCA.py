# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Load your dataset from a CSV file
df = pd.read_csv("Mall_Customers.csv")

# Display the first few rows of the dataset
print(df.head())
print("\nDataset shape:", df.shape)

# Preprocessing: Drop CustomerID and Gender - Those are unneeded for PCA as CustomerID is just an ID and Gender is a binary value
df = df.drop(columns=['CustomerID', 'Gender'])

# Step 1: Separate features for PCA
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Step 2: Standardize the features - We standardize the features to have a mean of zero and a standard deviation of one using StandardScaler. 
# This step ensures that all features contribute equally to the PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled data shape:", X_scaled.shape)

# Step 3: Calculate Covariance Matrix
# Then calculate eigenvectors and corresponding eigenvalues from the Covariance Matrix
cov_matrix = np.cov(X_scaled.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Number of eigenvectors:", len(eigenvectors))
print("Number of eigenvalues:", len(eigenvalues))

# Step 4: Sort eigenvectors according to their eigenvalues in decreasing order
# This is done to prioritize those that explain more variance
eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

print("Eigenvalues in descending order:")
for i in eig_pairs:
    print(i[0])

# Step 5: Calculate the percentage of variance explained by each principal component
total_variance = sum(eigenvalues)
variance_explained = [(i / total_variance) * 100 for i in sorted(eigenvalues, reverse=True)]

# Print variance explained for PC1, PC2, and PC3
for i in range(3):
    print(f"Variance explained by PC{i+1}: {variance_explained[i]:.2f}%")

# Print cumulative variance explained
cumulative_variance = np.cumsum(variance_explained[:3])
print(f"\nCumulative variance explained by first 3 PCs: {cumulative_variance[-1]:.2f}%")

# Step 6: Choose first k eigenvectors based on PC1, PC2 and PC3
# k = 3 is chosen since cumulative variance explained by k = 3 is 100%
k = 3
top_k_eigenvectors = np.array([eig_pairs[i][1] for i in range(k)])

print("Shape of selected eigenvectors:", top_k_eigenvectors.shape)

# Step 7: Transform the original n-dimensional data points into k dimensions
# The original data is projected onto the new principal component using matrix multiplication
X_pca = X_scaled.dot(top_k_eigenvectors.T)
print("Shape of transformed data:", X_pca.shape)

# Step 8: Plot the final PCA representation in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c='b', label='Customers')
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Third Principal Component')
ax.set_title('3D PCA Representation of Mall Customers Dataset (k=3)')
ax.legend()
plt.show()
