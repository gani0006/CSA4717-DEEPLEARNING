import numpy as np

# Define the function
def f(x):
    return x**2

# Define the gradient of the function
def grad_f(x):
    return 2 * x

# Gradient descent function
def gradient_descent(initial_x, learning_rate, num_iterations):
    x = initial_x
    for _ in range(num_iterations):
        gradient = grad_f(x)
        x = x - learning_rate * gradient
    return x

# Initial guess
initial_x = 5.0

# Learning rate
learning_rate = 0.1

# Number of iterations
num_iterations = 100

# Run gradient descent
min_x = gradient_descent(initial_x, learning_rate, num_iterations)

print("Minimum value of x:", min_x)
print("Minimum value of f(x):", f(min_x))
