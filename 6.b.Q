from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # Common Naive Bayes implementation for Iris
from sklearn.metrics import accuracy_score

# Load the Iris dataset (same as KNN)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable

# Split data into training and testing sets (same as KNN)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the Naive Bayes model
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", accuracy)
