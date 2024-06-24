from utils import load_mnist
from neural_network import NeuralNetwork
import numpy as np

if __name__ == "__main__":
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    data_dir = 'data'
    (X_train, y_train), (X_test, y_test) = load_mnist(data_dir)

    # Rest of the code remains the same
    input_size = X_train.shape[1]  # 28*28 = 784
    hidden_size = 128
    output_size = 10  # 10 digits

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Normalize the data
    X_train /= 255.0
    X_test /= 255.0

    # Split training data to create a validation set
    validation_split = 0.1
    split_idx = int(len(X_train) * (1 - validation_split))
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]

    # Initialize and train the network
    print("Training the network...")
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X_train, y_train, epochs=100, batch_size=32, learning_rate=0.01, 
             validation_data=(X_val, y_val), patience=10)

    # Evaluate on test set
    test_accuracy = np.mean(nn.predict(X_test) == y_test)
    print(f'Test accuracy: {test_accuracy}')

    # Save the trained model
    model_filename = 'mnist_model.pkl'
    nn.save(model_filename)
    print(f"Model saved to {model_filename}")