import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training the perceptron
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # Update weights and bias
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = [self.activation_function(x) for x in linear_output]
        return np.array(y_predicted)

# Load iris dataset
iris = datasets.load_iris()
X = iris.data[:100, :2]  # Use only the first two features for simplicity
y = iris.target[:100]     # Use only the first two classes (0 and 1)

# Create and train the perceptron
perceptron = Perceptron(learning_rate=0.1, n_iters=100)
perceptron.fit(X, y)

# Predictions
predictions = perceptron.predict(X)

# Plotting the results
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', label='Data points')
plt.scatter(X[predictions == 1][:, 0], X[predictions == 1][:, 1], color='red', label='Predicted Class 1')
plt.scatter(X[predictions == 0][:, 0], X[predictions == 0][:, 1], color='blue', label='Predicted Class 0')
plt.title('Perceptron Classification on Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# Print weights and bias
print(f"Weights: {perceptron.weights}")
print(f"Bias: {perceptron.bias}")
