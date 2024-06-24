import os
import struct
import numpy as np

# Utility functions for loading MNIST data
def load_mnist_images(filepath):
    with open(filepath, 'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape((size, nrows * ncols))
    return data.astype(np.float32) / 255.0

def load_mnist_labels(filepath):
    with open(filepath, 'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

def load_mnist(path='.'):
    train_images = load_mnist_images(os.path.join(path, 'train-images.idx3-ubyte'))
    train_labels = load_mnist_labels(os.path.join(path, 'train-labels.idx1-ubyte'))
    test_images = load_mnist_images(os.path.join(path, 't10k-images.idx3-ubyte'))
    test_labels = load_mnist_labels(os.path.join(path, 't10k-labels.idx1-ubyte'))
    return (train_images, train_labels), (test_images, test_labels)