import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split

from keras.datasets import fashion_mnist

# Load the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

class FFNeuralNetwork():
    def __init__(self, 
                neurons=128, 
                hid_layers=4, 
                input_size=784, 
                output_size=10, 
                act_func="sigmoid", 
                weight_init="random", 
                out_act_func="softmax",
                init_toggle=True):
                
        self.neurons, self.hidden_layers = neurons, hid_layers
        self.weights, self.biases = [], []
        self.input_size, self.output_size = input_size, output_size
        self.activation_function, self.weight_init = act_func, weight_init
        self.output_activation_function = out_act_func

        if init_toggle:
            self.initialize_weights()
            self.initiate_biases()

    def initialize_weights(self):
        self.weights.append(np.random.randn(self.input_size, self.neurons))
        for _ in range(self.hidden_layers - 1):
            self.weights.append(np.random.randn(self.neurons, self.neurons))
        self.weights.append(np.random.randn(self.neurons, self.output_size))

        if self.weight_init == "xavier":
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] / np.sqrt(self.weights[i].shape[0])

    def initiate_biases(self):
        for _ in range(self.hidden_layers):
            self.biases.append(np.zeros(self.neurons))
        self.biases.append(np.zeros(self.output_size))
    
    def activation(self, x):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "tanh":
            return np.tanh(x)
        elif self.activation_function == "ReLU":
            return np.maximum(1e-20, x)
        else:
            raise Exception("Invalid activation function")
    
    def output_activation(self, x):
        if self.output_activation_function == "softmax":
            max_x = np.max(x, axis=1)
            max_x = max_x.reshape(max_x.shape[0], 1)
            exp_x = np.exp(x - max_x)
            softmax_mat = exp_x / np.sum(exp_x, axis=1).reshape(exp_x.shape[0], 1)
            # change 0s to 1e-10
            softmax_mat[softmax_mat == 0] = 1e-20
            return softmax_mat
        else:
            raise Exception("Invalid output activation function")
    
    def forward(self, x):
        self.pre_activation, self.post_activation = [x], [x]

        for i in range(self.hidden_layers):
            self.pre_activation.append(np.matmul(self.post_activation[-1], self.weights[i]) + self.biases[i])
            self.post_activation.append(self.activation(self.pre_activation[-1]))
            
        self.pre_activation.append(np.matmul(self.post_activation[-1], self.weights[-1]) + self.biases[-1])
        self.post_activation.append(self.output_activation(self.pre_activation[-1]))

        return self.post_activation[-1]

def loss(loss, y, y_pred):
    if loss == "cross_entropy":
        return -np.sum(y * np.log(y_pred))
    elif loss == "mean_squared":
        return np.sum((y - y_pred) ** 2) / 2
    else:
        raise Exception("Invalid loss function")

class Backpropagation():
    def __init__(self, 
                 nn: FFNeuralNetwork, 
                 loss="cross_entropy", 
                 act_func="sigmoid"):
        
        self.nn, self.loss, self.activation_function = nn, loss, act_func
    
    def loss_derivative(self, y, y_pred):
        if self.loss == "cross_entropy":
            return -y / y_pred
        elif self.loss == "mse":
            return 2 * (y_pred - y)
        else:
            raise Exception("Invalid loss function")
        
    def activation_derivative(self, x):
        # x is the post-activation value
        if self.activation_function == "sigmoid":
            return x * (1 - x)
        elif self.activation_function == "tanh":
            return 1 - x ** 2
        elif self.activation_function == "ReLU":
            return 1 * (x > 0)
        else:
            raise Exception("Invalid activation function")
        
    def output_activation_derivative(self, y, y_pred):
        # x is the post-activation value
        if self.nn.output_activation_function == "softmax":
            return y_pred - y
        else:
            raise Exception("Invalid output activation function")
    
    def backward(self, y, y_pred):
        self.d_weights, self.d_biases = [], []
        self.d_h, self.d_a = [], []

        self.d_h.append(self.loss_derivative(y, y_pred))
        self.d_a.append(self.output_activation_derivative(y, y_pred))

        for i in range(self.nn.hidden_layers, 0, -1):
            self.d_weights.append(np.matmul(self.nn.post_activation[i].T, self.d_a[-1]))
            self.d_biases.append(np.sum(self.d_a[-1], axis=0))
            self.d_h.append(np.matmul(self.d_a[-1], self.nn.weights[i].T))
            self.d_a.append(self.d_h[-1] * self.activation_derivative(self.nn.post_activation[i]))

        self.d_weights.append(np.matmul(self.nn.post_activation[0].T, self.d_a[-1]))
        self.d_biases.append(np.sum(self.d_a[-1], axis=0))

        self.d_weights.reverse()
        self.d_biases.reverse()

        return self.d_weights, self.d_biases

class Optimiser():
    def __init__(self, 
                 nn: FFNeuralNetwork, 
                 bp:Backpropagation, 
                 lr=0.01, 
                 optimiser="sgd", 
                 momentum=0.9,
                 epsilon=1e-8,
                 beta1=0.9,
                 beta2=0.999, 
                 t=0,
                 decay=0):
        
        self.nn, self.bp, self.lr, self.optimiser = nn, bp, lr, optimiser
        self.momentum, self.epsilon, self.beta1, self.beta2 = momentum, epsilon, beta1, beta2
        self.h_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.h_biases = [np.zeros_like(b) for b in self.nn.biases]
        self.hm_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.hm_biases = [np.zeros_like(b) for b in self.nn.biases]
        self.t = t
        self.decay = decay

    def run(self, d_weights, d_biases, y, x):
        if(self.optimiser == "sgd"):
            self.SGD(d_weights, d_biases)
        elif(self.optimiser == "momentum"):
            self.MomentumGD(d_weights, d_biases)
        elif(self.optimiser == "nag"):
            self.NAG(d_weights, d_biases)
        elif (self.optimiser == "nag2"):
            self.NAG2(y, x)
        elif(self.optimiser == "rmsprop"):
            self.RMSProp(d_weights, d_biases)
        elif(self.optimiser == "adam"):
            self.Adam(d_weights, d_biases)
        elif (self.optimiser == "nadam"):
            self.NAdam(d_weights, d_biases)
        else:
            raise Exception("Invalid optimiser")
    
    def SGD(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.nn.weights[i] -= self.lr * (d_weights[i] + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (d_biases[i] + self.decay * self.nn.biases[i])

    def MomentumGD(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.h_weights[i] = self.momentum * self.h_weights[i] + d_weights[i]
            self.h_biases[i] = self.momentum * self.h_biases[i] + d_biases[i]

            self.nn.weights[i] -= self.lr * (self.h_weights[i] + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (self.h_biases[i] + self.decay * self.nn.biases[i])

    def NAG(self, d_weights, d_biases):        
        for i in range(self.nn.hidden_layers + 1):
            self.h_weights[i] = self.momentum * self.h_weights[i] + d_weights[i]
            self.h_biases[i] = self.momentum * self.h_biases[i] + d_biases[i]

            self.nn.weights[i] -= self.lr * (self.momentum * self.h_weights[i] + d_weights[i] + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (self.momentum * self.h_biases[i] + d_biases[i] + self.decay * self.nn.biases[i])

    def NAG2(self, y, x):
        nn_new = FFNeuralNetwork(neurons = self.nn.neurons,
                                 input_size = self.nn.input_size,
                                 output_size = self.nn.output_size,
                                 hid_layers = self.nn.hidden_layers,
                                 act_func = self.nn.activation_function,
                                 out_act_func = self.nn.output_activation_function,
                                 weight_init = self.nn.weight_init,
                                 init_toggle = False)
        
        bp_new = Backpropagation(nn = nn_new, 
                                 loss = self.bp.loss,
                                 act_func = self.bp.activation_function)
        
        nn_new.weights = [w - self.momentum * self.h_weights[i] for i, w in enumerate(self.nn.weights)]
        nn_new.biases = [b - self.momentum * self.h_biases[i] for i, b in enumerate(self.nn.biases)]

        y_pred_new = nn_new.forward(x)
        d_weights_new, d_biases_new = bp_new.backward(y, y_pred_new)

        for i in range(self.nn.hidden_layers + 1):
            self.h_weights[i] = self.momentum * self.h_weights[i] + d_weights_new[i]
            self.h_biases[i] = self.momentum * self.h_biases[i] + d_biases_new[i]

            self.nn.weights[i] -= self.h_weights[i] * self.lr
            self.nn.biases[i] -= self.h_biases[i] * self.lr

    def RMSProp(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.h_weights[i] = self.momentum * self.h_weights[i] + (1 - self.momentum) * d_weights[i]**2
            self.h_biases[i] = self.momentum * self.h_biases[i] + (1 - self.momentum) * d_biases[i]**2

            self.nn.weights[i] -= (self.lr / (np.sqrt(self.h_weights[i]) + self.epsilon)) * d_weights[i] + self.decay * self.nn.weights[i] * self.lr
            self.nn.biases[i] -= (self.lr / (np.sqrt(self.h_biases[i]) + self.epsilon)) * d_biases[i] + self.decay * self.nn.biases[i] * self.lr

    def Adam(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.hm_weights[i] = self.beta1 * self.hm_weights[i] + (1 - self.beta1) * d_weights[i]
            self.hm_biases[i] = self.beta1 * self.hm_biases[i] + (1 - self.beta1) * d_biases[i]

            self.h_weights[i] = self.beta2 * self.h_weights[i] + (1 - self.beta2) * d_weights[i]**2
            self.h_biases[i] = self.beta2 * self.h_biases[i] + (1 - self.beta2) * d_biases[i]**2

            self.hm_weights_hat = self.hm_weights[i] / (1 - self.beta1**(self.t + 1))
            self.hm_biases_hat = self.hm_biases[i] / (1 - self.beta1**(self.t + 1))

            self.h_weights_hat = self.h_weights[i] / (1 - self.beta2**(self.t + 1))
            self.h_biases_hat = self.h_biases[i] / (1 - self.beta2**(self.t + 1))

            self.nn.weights[i] -= self.lr * (self.hm_weights_hat / ((np.sqrt(self.h_weights_hat)) + self.epsilon)) + self.decay * self.nn.weights[i] * self.lr
            self.nn.biases[i] -= self.lr * (self.hm_biases_hat / ((np.sqrt(self.h_biases_hat)) + self.epsilon)) + self.decay * self.nn.biases[i] * self.lr

    def NAdam(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.hm_weights[i] = self.beta1 * self.hm_weights[i] + (1 - self.beta1) * d_weights[i]
            self.hm_biases[i] = self.beta1 * self.hm_biases[i] + (1 - self.beta1) * d_biases[i]

            self.h_weights[i] = self.beta2 * self.h_weights[i] + (1 - self.beta2) * d_weights[i]**2
            self.h_biases[i] = self.beta2 * self.h_biases[i] + (1 - self.beta2) * d_biases[i]**2

            self.hm_weights_hat = self.hm_weights[i] / (1 - self.beta1 ** (self.t + 1))
            self.hm_biases_hat = self.hm_biases[i] / (1 - self.beta1 ** (self.t + 1))

            self.h_weights_hat = self.h_weights[i] / (1 - self.beta2 ** (self.t + 1))
            self.h_biases_hat = self.h_biases[i] / (1 - self.beta2 ** (self.t + 1))

            temp_update_w = self.beta1 * self.hm_weights_hat + ((1 - self.beta1) / (1 - self.beta1 ** (self.t + 1))) * d_weights[i]
            temp_update_b = self.beta1 * self.hm_biases_hat + ((1 - self.beta1) / (1 - self.beta1 ** (self.t + 1))) * d_biases[i]

            self.nn.weights[i] -= self.lr * (temp_update_w / ((np.sqrt(self.h_weights_hat)) + self.epsilon)) + self.decay * self.nn.weights[i] * self.lr
            self.nn.biases[i] -= self.lr * (temp_update_b / ((np.sqrt(self.h_biases_hat)) + self.epsilon)) + self.decay * self.nn.biases[i] * self.lr

import wandb
wandb.login()
wandb.WANDB_NOTEBOOK_NAME = "Assignment1.ipynb"

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'val_accuracy'
    },
    'parameters': {
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'learning_rate': {
            'values': [0.0001, 0.001, 0.01, 0.1]
        },
        'neurons': {
            'values': [16, 32, 64, 128]
        },
        'hidden_layers': {
            'values': [1, 2, 3, 4]
        },
        'activation': {
            'values': ['ReLU', 'tanh', 'sigmoid']
        },
        'weight_init': {
            'values': ['Xavier', 'random']
        },
        'optimiser': {
            'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
        },
        'momentum': {
            'values': [0.8, 0.9]
        },
        'input_size': {
            'value': 784
        },
        'output_size': {
            'value': 10
        },
        'loss': {
            'value': 'cross_entropy'
        },
        'epochs': {
            'value': 10
        },
        'beta1': {
            'value': 0.9
        },
        'beta2': {
            'value': 0.999
        },
        'output_activation': {
            'value': 'softmax'
        },
        'epsilon': {
            'value': 1e-8
        },
        'decay': {
            'values': [0, 0.1, 0.01, 0.001]
        }
    }
}

def load_data(type):
    (x, y), (x_test, y_test) = fashion_mnist.load_data()

    if type == 'train':
        x = x.reshape(x.shape[0], 784)
        x = x.astype('float32')
        x /= 255
        # change y to one hot
        y = np.eye(10)[y]
        return x, y
    elif type == 'test':
        x_test = x_test.reshape(x_test.shape[0], 784)
        x_test = x_test.astype('float32')
        x_test /= 255
        # change y to one hot
        y_test = np.eye(10)[y_test]
        return x_test, y_test

def train():
    x_train, y_train = load_data('train')

    run = wandb.init()
    parameters = wandb.config
    run.name = f"{parameters['activation']}_neurons={parameters['neurons']}_layers={parameters['hidden_layers']}_lr={parameters['learning_rate']}_batch={parameters['batch_size']}_opt={parameters['optimiser']}_mom={parameters['momentum']}_init={parameters['weight_init']}"
    
    nn = FFNeuralNetwork(input_size=parameters['input_size'], 
                         hid_layers=parameters['hidden_layers'], 
                         neurons=parameters['neurons'], 
                         output_size=parameters['output_size'], 
                         act_func=parameters['activation'], 
                         out_act_func=parameters['output_activation'],
                         weight_init=parameters['weight_init'])
    bp = Backpropagation(nn=nn, 
                         loss=parameters['loss'],
                         act_func=parameters['activation'])
    opt = Optimiser(nn=nn,
                    bp=bp,
                    lr=parameters['learning_rate'],
                    optimiser=parameters['optimiser'],
                    momentum=parameters['momentum'],
                    epsilon=parameters['epsilon'],
                    beta1=parameters['beta1'],
                    beta2=parameters['beta2'],
                    decay=parameters['decay'])
    
    batch_size = parameters['batch_size']
    x_train_act, x_val, y_train_act, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    print("Initial Accuracy: {}".format(np.sum(np.argmax(nn.forward(x_train), axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]))

    for epoch in range(parameters['epochs']):
        for i in range(0, x_train_act.shape[0], batch_size):
            x_batch = x_train_act[i:i+batch_size]
            y_batch = y_train_act[i:i+batch_size]

            y_pred = nn.forward(x_batch)
            d_weights, d_biases = bp.backward(y_batch, y_pred)
            opt.run(d_weights, d_biases, y_batch, x_batch)
        
        opt.t += 1

        y_pred = nn.forward(x_train_act)
        print("Epoch: {}, Loss: {}".format(epoch + 1, loss("cross_entropy", y_train_act, y_pred)))
        print("Accuracy: {}".format(np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train_act, axis=1)) / y_train_act.shape[0]))

        train_loss = loss("cross_entropy", y_train_act, y_pred)
        train_accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train_act, axis=1)) / y_train_act.shape[0]
        val_loss = loss("cross_entropy", y_val, nn.forward(x_val))
        val_accuracy = np.sum(np.argmax(nn.forward(x_val), axis=1) == np.argmax(y_val, axis=1)) / y_val.shape[0]

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
    
    return nn

wandb_id = wandb.sweep(sweep_configuration, project="sweep-hyperparameters-new")

wandb.agent(wandb_id, function=train, count=500)