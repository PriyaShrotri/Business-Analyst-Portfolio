import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Load the dataset from an Excel file
df = pd.read_csv("Reviews.csv",  usecols=['Score', 'Summary', 'Text'])

# Convert the dataframe to a list of lists for easier manipulation
dataset = df.values.tolist()

# Display the first 5 rows after preprocessing
print("Dataset after preprocessing (first 5 rows):")
print(df.head())
print("\n")

# Plot the distribution of actual scores before analysis
plt.figure(figsize=(10, 6))
df['Score'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Review Scores')
plt.xlabel('Review Score')
plt.ylabel('Count of reviews')
plt.show()

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment scores and scale them to 0-5
def get_scaled_sentiment_scores(text):
    compound_score = sia.polarity_scores(text)['compound']
    # Scale from [-1, 1] to [0, 5]
    return (compound_score + 1) * 2.5

# Apply sentiment analysis to the 'Text' column
df['sentiment_score'] = df['Text'].apply(get_scaled_sentiment_scores)
print("\nSentiment analysis completed.")
print("\nFirst few rows with sentiment scores:")
print(df[['Text', 'Score', 'sentiment_score']].head())

# Plot the distribution of sentiment scores
plt.figure(figsize=(10, 6))
plt.hist(df['sentiment_score'], bins=50, edgecolor='black')
plt.title('Distribution of Sentiment Scores (0-5 scale)')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.show()

# Prepare data for linear regression
X = df[['sentiment_score']]
y = df['Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split into training and testing sets.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nLinear regression model trained.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("\nModel Evaluation Results:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Plot actual vs predicted scores
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Scores')
plt.tight_layout()
plt.show()

# Print correlation between sentiment scores and actual scores
correlation = df['sentiment_score'].corr(df['Score'])
print(f"\nCorrelation between sentiment scores and actual scores: {correlation}")
