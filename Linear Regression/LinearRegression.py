import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the Salary dataset
data = pd.read_csv("Salary_Data.csv")
x = data['YearsExperience'].values.reshape(-1, 1)
y = data['Salary'].values

# Create a scatter plot of the data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Years of Experience')

# Create and fit the linear regression model
model = LinearRegression()
model.fit(x, y)

# Print model coefficients
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.2f}")

# Calculate R-squared
r_squared = model.score(x, y)
print(f"R-squared: {r_squared:.2f}")

# Make predictions using the trained model
y_pred = model.predict(x)

# Visualize the regression line with the data
plt.plot(x, y_pred, color='red', label='Regression line')
plt.legend()
plt.show()
