import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Initialize a logistic regression model
model = LogisticRegression(max_iter=1000)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

# Print the cross-validation scores
print("Cross-validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))
