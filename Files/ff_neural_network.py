import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Files.settings as settings

# A feedforward neural network which takes images from the fashion-mnist 
# data as input and outputs a probability distribution over the 10 classes.

# The network has 3 layers: input, hidden, and output.
# The input layer is a 784-dimensional vector, where each dimension corresponds
# to a pixel in the image, the output layer is of size 10


class FFNeuralNetwork():
    def __init__(self, neurons, hidden_layers, input_size, output_size, activation_function):
        self.neurons = neurons
        self.hidden_layers = hidden_layers
        self.weights = []
        self.biases = []
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

    def initialize_weights(self):
        for i in range(self.hidden_layers + 1):
            if i == 0:
                self.weights.append(np.random.randn(self.neurons, self.input_size))
                self.biases.append(np.random.randn(self.neurons, 1))
            elif i == self.hidden_layers:
                self.weights.append(np.random.randn(self.output_size, self.neurons))
                self.biases.append(np.random.randn(self.output_size, 1))
            else:
                self.weights.append(np.random.randn(self.neurons, self.neurons))
                self.biases.append(np.random.randn(self.neurons, 1))
    
    


