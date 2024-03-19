#7(b)

import numpy as np

# Generate modified sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 2 * X**2 + 3 * X + 1 + np.random.randn(100, 1)  # Quadratic function instead of linear

# Number of iterations, learning rate, and stopping threshold
iterations = 1000
learning_rate = 0.01
stopping_threshold = 0.0001

# Function to calculate the mean squared error (loss)
def compute_loss(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    return np.sum(error ** 2) / (2 * m)

# Gradient Descent Function
def gradient_descent(X, y, theta, learning_rate, iterations, stopping_threshold):
    m = len(y)
    loss_history = []

    for i in range(iterations):
        gradient = X.T.dot(X.dot(theta) - y) / m
        theta = theta - learning_rate * gradient
        loss = compute_loss(X, y, theta)
        loss_history.append(loss)

        # Stopping criteria
        if len(loss_history) > 1 and np.abs(loss_history[-2] - loss_history[-1]) < stopping_threshold:
            break

    return theta, loss_history

# Add bias term and quadratic term to X
X_b = np.c_[np.ones((len(X), 1)), X, X**2]

# Initial guess for parameters
theta_initial = np.random.randn(3, 1)

# Run gradient descent
theta_optimal, loss_history = gradient_descent(X_b, y, theta_initial, learning_rate, iterations, stopping_threshold)

print("Optimal Parameters (theta):", theta_optimal)

# Plot the loss curve
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Gradient Descent: Loss over Iterations')
plt.show()
