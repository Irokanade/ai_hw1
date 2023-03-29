from sklearn.preprocessing import MinMaxScaler
import numpy as np

class MyMinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min = None
        self.max = None
    
    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self
    
    def transform(self, X):
        X_scaled = (X - self.min) / (self.max - self.min)
        X_scaled = X_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return X_scaled
    
    def inverse_transform(self, X_scaled):
        X = (X_scaled - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        X = X * (self.max - self.min) + self.min
        return X

class NeuralNetwork:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with random values
        self.weights1 = 2 * np.random.randn(self.input_size, self.hidden_size) - 1
        self.weights2 = 2 * np.random.randn(self.hidden_size, self.output_size) - 1

        # Initialize biases with random values
        self.bias1 = 2 * np.random.randn(1, self.hidden_size) - 1
        self.bias2 = 2 * np.random.randn(1, self.output_size) - 1

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def feedforward(self, X):
        # Feedforward calculation
        self.layer1 = self.tanh(np.dot(X, self.weights1) + self.bias1)
        self.output = self.tanh(np.dot(self.layer1, self.weights2) + self.bias2)
        
    def backpropagation(self, X, y, learning_rate):
        # Backpropagation calculation
        output_error = y - self.output
        output_delta = output_error * self.tanh_derivative(self.output)
        layer1_error = np.dot(output_delta, self.weights2.T)
        layer1_delta = layer1_error * self.tanh_derivative(self.layer1)
        self.weights2 += learning_rate * np.dot(self.layer1.T, output_delta)
        self.weights1 += learning_rate * np.dot(X.T, layer1_delta)
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.bias1 += np.sum(layer1_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, learning_rate, epochs, print_error_every=100):
        # Train the neural network for a fixed number of iterations
        for i in range(epochs):
            self.feedforward(X)
            self.backpropagation(X, y, learning_rate)
            if (i+1) % print_error_every == 0:
                mse = np.mean((y - self.predict(X))**2)
                print(f"Epoch {i+1} - Mean Squared Error: {mse:.6f}")
            
    def predict(self, X):
        # Make a prediction for a new input value
        self.feedforward(X)
        return self.output


train_in = np.array([
    [11,	86,	90,	93.5,	93,	90.63,	90,	99,	106,	98.33],
    [11,	85.5,	70,	86.5,	88.5,	82.63,	91,	88,	90,	89.67],
    [10,	77,	63.5,	58.5,	78,	69.25,	82,	73,	65,	7.33],
    #[8,	82.5,	75.5,	87.5,	82.5,	82.00,	75,	99,	91,	88.33],
    [11,	91,	89,	88.5,	89.75,	89.65,	105,	97,	114,	105.33],
    #[11,	94.5,	89.5,	95.5,	90.5,	92.50,	93,	116,	112,	107.00],
    [6,	78.25,	63.5,	62.5,	67.25,	67.88,	94,	66,	77,	79.00],
    #[5,	86.5,	90.5,	82.5,	0,	64.88,	85,	94,	90,	89.67],
    [11,	85,	62,	63,	84.5,	73.63,	96,	95,	74,	88.33],
    #[10,	89,	77.5,	90,	90.5,	86.75,  95,	105,	97,	99.00],
    [8,	91.25,	90.5,	88.5,	0,	67.56,	83,	83,	60,	75.33],
    [7,	70.5,	61.5,	59,	52,	60.75,	98,	95,	38,	77.00],
    [7,	84,	0,	0,	0,	21.00,	65,	0,	0,	21.67],
    [11,	64,	9,	0,	0,	18.25,	85,	48,	71,	68.00]
])

# train_sol = np.array([[94, 89, 79, 85, 95, 96, 75, 75, 85, 91, 76, 73, 44, 57]]).T
train_sol = np.array([[94, 89, 79, 95, 75, 85, 76, 73, 44, 57]]).T
predict_in = np.array([
    [8,	82.5,	75.5,	87.5,	82.5,	82.00,	75,	99,	91,	88.33],
    [11,	94.5,	89.5,	95.5,	90.5,	92.50,	93,	116,	112,	107.00],
    [5,	86.5,	90.5,	82.5,	0,	64.88,	85,	94,	90,	89.67],
    [10,	89,	77.5,	90,	90.5,	86.75,  95,	105,	97,	99.00],
])

# Scale training input
scaler_x = MinMaxScaler(feature_range=(-1, 1))
train_in_scaled = scaler_x.fit_transform(train_in)

# Scale training solution
scaler_y = MinMaxScaler(feature_range=(-1, 1))
train_sol_scaled = scaler_y.fit_transform(train_sol)

# Scale prediction input
predict_in_scaled = scaler_x.transform(predict_in)

# print(train_in_scaled)
# print(train_sol_scaled)
# print(predict_in_scaled)

nn = NeuralNetwork(input_size=10, hidden_size=1, output_size=1)

nn.train(train_in_scaled, train_sol_scaled, learning_rate=0.1, epochs=1000000)
print('Output')
output_rescaled = scaler_y.inverse_transform(nn.output)
print(output_rescaled)

print('Predict')
nn.predict(predict_in_scaled)
predict_out_scaled = scaler_y.inverse_transform(nn.output)
print(predict_out_scaled)
# predict_output = scaler.inverse_transform(nn.output)
# print('New output')
# print(predict_output)