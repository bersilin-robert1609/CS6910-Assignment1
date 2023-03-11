'''
Bersilin C | CS20B013
CS6910: Assignment 1
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist, mnist
import wandb
import argparse

# Class names for the Fashion MNIST dataset
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Settings for the run
WANDB_PROJECT = "myprojectname"
WANDB_ENTITY = "myname"

# The below are my best parameters for the Fashion MNIST dataset
DATASET = "fashion_mnist"
EPOCHS = 10
BATCH_SIZE = 64
LOSS = "cross_entropy"
OPTIMIZER = "nadam"
LEARNING_RATE = 0.005
MOMENTUM = 0.8
BETA = 0.9
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
WEIGHT_DECAY = 0.0005
WEIGHT_INIT = "xavier"
NUM_LAYERS = 1
HIDDEN_SIZE = 256
ACTIVATION = "sigmoid"
INPUT_SIZE = 784
OUTPUT_SIZE = 10

parameters_dict = {
    "wandb_project": WANDB_PROJECT,
    "wandb_entity": WANDB_ENTITY,
    "dataset": DATASET,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "loss": LOSS,
    "optimizer": OPTIMIZER,
    "learning_rate": LEARNING_RATE,
    "momentum": MOMENTUM,
    "beta": BETA,
    "beta1": BETA1,
    "beta2": BETA2,
    "epsilon": EPSILON,
    "weight_decay": WEIGHT_DECAY,
    "weight_init": WEIGHT_INIT,
    "num_layers": NUM_LAYERS,
    "hidden_size": HIDDEN_SIZE,
    "activation": ACTIVATION
}

# Parse arguments and update parameters_dict
parser = argparse.ArgumentParser()
parser.add_argument("-wp", "--wandb_project", type=str, default=WANDB_PROJECT, help="Wandb project name", required=True)
parser.add_argument("-we", "--wandb_entity", type=str, default=WANDB_ENTITY, help="Wandb entity name", required=True)
parser.add_argument("-d", "--dataset", type=str, default=DATASET, help="Dataset to use choices=['fashion_mnist', 'mnist']")
parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="Number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
parser.add_argument("-l", "--loss", type=str, default=LOSS, help="Loss function to use choices=['cross_entropy', 'mean_squared_error']")
parser.add_argument("-o", "--optimizer", type=str, default=OPTIMIZER, help="Optimizer to use choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']")
parser.add_argument("-lr", "--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate")
parser.add_argument("-m", "--momentum", type=float, default=MOMENTUM, help="Momentum for Momentum and NAG")
parser.add_argument("-beta", "--beta", type=float, default=BETA, help="Beta for RMSProp")
parser.add_argument("-beta1", "--beta1", type=float, default=BETA1, help="Beta1 for Adam and Nadam")
parser.add_argument("-beta2", "--beta2", type=float, default=BETA2, help="Beta2 for Adam and Nadam")
parser.add_argument("-eps", "--epsilon", type=float, default=EPSILON, help="Epsilon for Adam and Nadam")
parser.add_argument("-w_d", "--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
parser.add_argument("-w_i", "--weight_init", type=str, default=WEIGHT_INIT, help="Weight initialization choices=['random', 'xavier']")
parser.add_argument("-nhl", "--num_layers", type=int, default=NUM_LAYERS, help="Number of hidden layers")
parser.add_argument("-sz", "--hidden_size", type=int, default=HIDDEN_SIZE, help="Hidden size")
parser.add_argument("-a", "--activation", type=str, default=ACTIVATION, help="Activation function choices=['sigmoid', 'tanh', 'relu']")

args = parser.parse_args()
parameters_dict.update(vars(args))

# Print the parameters
print("Parameters:")
for key, value in parameters_dict.items():
    print(f"{key}: {value}")

# Feedforward neural network
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
                self.weights[i] = self.weights[i] * np.sqrt(1 / self.weights[i].shape[0])
        
        if(self.weight_init != "random" and self.weight_init != "xavier"):
            raise Exception("Invalid weight initialization method")

    def initiate_biases(self):
        for _ in range(self.hidden_layers):
            self.biases.append(np.zeros(self.neurons))
        self.biases.append(np.zeros(self.output_size))
    
    def activation(self, x):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "tanh":
            return np.tanh(x)
        elif self.activation_function == "relu":
            return np.maximum(0, x)
        elif self.activation_function == "identity":
            return x
        else:
            raise Exception("Invalid activation function")
    
    def output_activation(self, x):
        if self.output_activation_function == "softmax":
            max_x = np.max(x, axis=1)
            max_x = max_x.reshape(max_x.shape[0], 1)
            exp_x = np.exp(x - max_x)
            softmax_mat = exp_x / np.sum(exp_x, axis=1).reshape(exp_x.shape[0], 1)
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
    
# Loss functions
def loss(loss, y, y_pred):
    if loss == "cross_entropy": # Cross Entropy
        return -np.sum(y * np.log(y_pred))
    elif loss == "mean_squared_error": # Mean Squared Error
        return np.sum((y - y_pred) ** 2) / 2
    else:
        raise Exception("Invalid loss function")
    
# Backpropagation
class Backpropagation():
    def __init__(self, 
                 nn: FFNeuralNetwork, 
                 loss="cross_entropy", 
                 act_func="sigmoid"):
        
        self.nn, self.loss, self.activation_function = nn, loss, act_func
    
    def loss_derivative(self, y, y_pred):
        if self.loss == "cross_entropy":
            return -y / y_pred
        elif self.loss == "mean_squared_error":
            return (y_pred - y)
        else:
            raise Exception("Invalid loss function")
        
    def activation_derivative(self, x):
        # x is the post-activation value
        if self.activation_function == "sigmoid":
            return x * (1 - x)
        elif self.activation_function == "tanh":
            return 1 - x ** 2
        elif self.activation_function == "relu":
            return (x > 0).astype(int)
        elif self.activation_function == "identity":
            return np.ones(x.shape)
        else:
            raise Exception("Invalid activation function")
        
    def output_activation_derivative(self, y, y_pred):
        if self.nn.output_activation_function == "softmax":
            # derivative of softmax is a matrix
            return np.diag(y_pred) - np.outer(y_pred, y_pred)
        else:
            raise Exception("Invalid output activation function")

    def backward(self, y, y_pred):
        self.d_weights, self.d_biases = [], []
        self.d_h, self.d_a = [], []

        self.d_h.append(self.loss_derivative(y, y_pred))
        output_derivative_matrix = []
        for i in range(y_pred.shape[0]):
            output_derivative_matrix.append(np.matmul(self.loss_derivative(y[i], y_pred[i]), self.output_activation_derivative(y[i], y_pred[i])))
        self.d_a.append(np.array(output_derivative_matrix))

        for i in range(self.nn.hidden_layers, 0, -1):
            self.d_weights.append(np.matmul(self.nn.post_activation[i].T, self.d_a[-1]))
            self.d_biases.append(np.sum(self.d_a[-1], axis=0))
            self.d_h.append(np.matmul(self.d_a[-1], self.nn.weights[i].T))
            self.d_a.append(self.d_h[-1] * self.activation_derivative(self.nn.post_activation[i]))

        self.d_weights.append(np.matmul(self.nn.post_activation[0].T, self.d_a[-1]))
        self.d_biases.append(np.sum(self.d_a[-1], axis=0))

        self.d_weights.reverse()
        self.d_biases.reverse()
        for i in range(len(self.d_weights)):
            self.d_weights[i] = self.d_weights[i] / y.shape[0]
            self.d_biases[i] = self.d_biases[i] / y.shape[0]

        return self.d_weights, self.d_biases
    
# Optimizers
class Optimizer():
    def __init__(self, 
                 nn: FFNeuralNetwork, 
                 bp:Backpropagation, 
                 lr=0.001, 
                 optimizer="sgd", 
                 momentum=0.9,
                 epsilon=1e-8,
                 beta=0.9,
                 beta1=0.9,
                 beta2=0.999, 
                 t=0,
                 decay=0):
        
        self.nn, self.bp, self.lr, self.optimizer = nn, bp, lr, optimizer
        self.momentum, self.epsilon, self.beta1, self.beta2, self.beta = momentum, epsilon, beta1, beta2, beta
        self.h_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.h_biases = [np.zeros_like(b) for b in self.nn.biases]
        self.hm_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.hm_biases = [np.zeros_like(b) for b in self.nn.biases]
        self.t = t
        self.decay = decay

    def run(self, d_weights, d_biases):
        if(self.optimizer == "sgd"):
            self.SGD(d_weights, d_biases)
        elif(self.optimizer == "momentum"):
            self.MomentumGD(d_weights, d_biases)
        elif(self.optimizer == "nag"):
            self.NAG(d_weights, d_biases)
        elif(self.optimizer == "rmsprop"):
            self.RMSProp(d_weights, d_biases)
        elif(self.optimizer == "adam"):
            self.Adam(d_weights, d_biases)
        elif (self.optimizer == "nadam"):
            self.NAdam(d_weights, d_biases)
        else:
            raise Exception("Invalid optimizer")
    
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

    def RMSProp(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.h_weights[i] = self.beta * self.h_weights[i] + (1 - self.beta) * d_weights[i]**2
            self.h_biases[i] = self.beta * self.h_biases[i] + (1 - self.beta) * d_biases[i]**2

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

# data loader function
def load_data(type, dataset=DATASET):
    x, y, x_test, y_test = None, None, None, None
    
    if dataset == 'mnist':
        (x, y), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x, y), (x_test, y_test) = fashion_mnist.load_data()

    if type == 'train':
        x_train = x.reshape(x.shape[0], 784) / 255
        y_train = np.eye(10)[y]
        return x_train, y_train
    elif type == 'test':
        x_test = x_test.reshape(x_test.shape[0], 784) / 255
        y_test = np.eye(10)[y_test]
        return x_test, y_test

# Initialize wandb
run = wandb.init(project=parameters_dict["wandb_project"], entity=parameters_dict["wandb_entity"], config=parameters_dict)

# Set name for the run ends with a random number
run.name = f"opt_{parameters_dict['optimizer']}_lr_{parameters_dict['learning_rate']}_act_{parameters_dict['activation']}_hid_{parameters_dict['num_layers']}_nrns_{parameters_dict['hidden_size']}_randn_" + str(np.random.randint(1000))

# Initialize the neural network
nn = FFNeuralNetwork(neurons=parameters_dict['hidden_size'],
                     hid_layers=parameters_dict['num_layers'],
                     input_size=INPUT_SIZE,
                     output_size=OUTPUT_SIZE,
                     act_func=parameters_dict['activation'],
                     weight_init=parameters_dict['weight_init'],
                     out_act_func="softmax",
                     init_toggle=True)

# Initialize the Backpropagation algorithm
bp = Backpropagation(nn=nn,
                     loss=parameters_dict['loss'],
                     act_func=parameters_dict['activation'])

# Initialize the optimizer
opt = Optimizer(nn=nn,
                bp=bp,
                lr=parameters_dict['learning_rate'],
                optimizer=parameters_dict['optimizer'],
                momentum=parameters_dict['momentum'],
                beta=parameters_dict['beta'],
                beta1=parameters_dict['beta1'],
                beta2=parameters_dict['beta2'],
                epsilon=parameters_dict['epsilon'],
                t=0,
                decay=parameters_dict['weight_decay'])

# Load the data
x_train, y_train = load_data(type='train', dataset=parameters_dict['dataset'])
x_test, y_test = load_data(type='test', dataset=parameters_dict['dataset'])

x_train_act, x_val, y_train_act, y_val = train_test_split(x_train, y_train, test_size=0.1)

# Initialize the training loop
for epoch in range(parameters_dict['epochs']):
    for i in range(0, x_train_act.shape[0], parameters_dict['batch_size']):
        x_batch = x_train_act[i:i + parameters_dict['batch_size']]
        y_batch = y_train_act[i:i + parameters_dict['batch_size']]

        # Forward pass
        y_pred = nn.forward(x_batch)

        # Backward pass
        d_weights, d_biases = bp.backward(y_batch, y_pred)

        # Update weights and biases
        opt.run(d_weights, d_biases)

    opt.t += 1

    y_pred = nn.forward(x_train_act)

    train_loss = loss(parameters_dict["loss"], y_train_act, y_pred)
    train_accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train_act, axis=1)) / y_train_act.shape[0]
    val_loss = loss(parameters_dict["loss"], y_val, nn.forward(x_val))
    val_accuracy = np.sum(np.argmax(nn.forward(x_val), axis=1) == np.argmax(y_val, axis=1)) / y_val.shape[0]

    print("Epoch: {}".format(epoch + 1))
    print("Train Accuracy: {}".format(train_accuracy))
    print("Validation Accuracy: {}".format(val_accuracy))
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

y_pred_test = nn.forward(x_test)
test_loss = loss(parameters_dict["loss"], y_test, y_pred_test)
test_accuracy = np.sum(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]

print("Test Accuracy: {}".format(test_accuracy))

wandb.log({
    "test_loss": test_loss,
    "test_accuracy": test_accuracy
})
wandb.finish()