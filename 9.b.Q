import numpy as np

class NeuralNetwork:
  def __init__(self, input_size, hidden_size, output_size):
    # Initialize weights with random values
    self.w1 = np.random.rand(input_size, hidden_size)
    self.b1 = np.zeros((1, hidden_size))
    self.w2 = np.random.rand(hidden_size, output_size)
    self.b2 = np.zeros((1, output_size))

  def sigmoid(self, x):
    # SIGMOID activation function
    return 1 / (1 + np.exp(-x))

  def predict(self, X):
    # Forward pass with SIGMOID activation
    z1 = X.dot(self.w1) + self.b1
    a1 = self.sigmoid(z1)
    z2 = a1.dot(self.w2) + self.b2
    a2 = self.sigmoid(z2)
    return a2

# Example usage
nn = NeuralNetwork(2, 4, 1)  # 2 input features, 4 hidden neurons, 1 output

# Sample input data
X = np.array([[0.1, 0.2], [0.7, 0.9]])

# Get network prediction
y_predicted = nn.predict(X)

print("Predicted output:", y_predicted)
