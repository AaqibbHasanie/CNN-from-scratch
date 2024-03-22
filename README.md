# Simple CNN Implementation for Handwritten Digit Classification from scratch

This project implements a Convolutional Neural Network (CNN) from scratch to classify handwritten digits. The CNN architecture is designed to work with 8x8 single-channel images with pixel values normalized between 0 and 1. This projects achieves an accuracy of **93%** on the test set

## Getting Started

Before running the project, ensure that you have all the required Python packages installed. You can install them by running:


pip install -r requirements.txt

# Project Structure

The project consists of the following files:

- `activation.py`: Contains implementations of various activation functions.
- `layers.py`: Defines the layers used in the CNN architecture, including fully connected layers and convolutional layers.
- `loss.py`: Implements the cross-entropy loss function.
- `main.py`: Main script to load the dataset, train the model, and evaluate its performance.
- `utils.py`: Utility functions for data loading, plotting, etc.
- `model.py`: Defines the CNN model architecture.
- `requirements.txt`: List of required Python packages.

## Implementation Details

### `layers.py`

- `FullyConnectedLayer`: Implements a fully connected neural network layer.
  - `__init__(self, in_features, out_features)`: Initializes the layer with the specified number of input and output neurons.
  - `forward(self, x)`: Performs forward pass computation.

- `Conv2dLayer`: Implements a 2D convolutional layer.
  - `__init__(self, in_channels, out_channels, kernel_size, stride)`: Initializes the layer with the specified parameters.
  - `forward(self, x)`: Performs forward pass computation.

### `activation.py`

Contains implementations of various activation functions, including:

- `ReLU`
- `Sigmoid`
- `Tanh`

Each activation function class provides the following methods:
- `__call__(self, x)`: Computes the activation function.
- `derivative(self, x)`: Computes the derivative of the activation function.

### `loss.py`

- `CrossEntropyLoss`: Implements the cross-entropy loss function.

## Usage

To train the model and evaluate its performance, run `main.py`:

```bash
python main.py
```

## Results

With everything implemented accurately, you should be able to receive an accuracy of 93% on the test set.
