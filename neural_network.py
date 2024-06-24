import numpy as np
from activation_function import relu, relu_derivative, softmax, he_initialization
import pickle

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, reg_lambda=0.01):
        self.W1 = he_initialization((input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = he_initialization((hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
        self.reg_lambda = reg_lambda

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate):
        batch_size = X.shape[0]
        delta2 = output.copy()
        delta2[range(batch_size), y] -= 1
        delta2 /= batch_size

        dW2 = np.dot(self.a1.T, delta2) + self.reg_lambda * self.W2
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = np.dot(delta2, self.W2.T) * relu_derivative(self.z1)
        dW1 = np.dot(X.T, delta1) + self.reg_lambda * self.W1
        db1 = np.sum(delta1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def update_weights(self, X, y, learning_rate):
        output = self.forward(X)
        self.backward(X, y, output, learning_rate)
        
    def train(self, X, y, epochs, batch_size, learning_rate, validation_data=None, patience=10):
        X_val, y_val = validation_data if validation_data else (None, None)
        best_accuracy = 0
        epochs_no_improve = 0

        for epoch in range(epochs):
            # Shuffle training data for each epoch
            shuffle_index = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[shuffle_index], y[shuffle_index]

            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                self.update_weights(X_batch, y_batch, learning_rate)

            if epoch % 10 == 0:
                train_accuracy = np.mean(np.argmax(self.forward(X), axis=1) == y)
                print(f'Epoch {epoch}, Training Accuracy: {train_accuracy}')

                if validation_data:
                    val_accuracy = np.mean(np.argmax(self.forward(X_val), axis=1) == y_val)
                    print(f'Epoch {epoch}, Validation Accuracy: {val_accuracy}')
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print('Early stopping.')
                        break

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)