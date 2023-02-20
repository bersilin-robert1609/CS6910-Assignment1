# set_parameters.py

# Description: This file sets all the paramaters from the project

# The settings are :
# WANDB_PROJECT - the name of the project in wandb
# WANDB_ENTITY - the name of the entity in wandb
# DATASET - the name of the dataset to use
# EPOCHS - the number of epochs to train for
# BATCH_SIZE - the batch size
# LOSS - the loss function to use
# OPTIMIZER - the optimizer to use
# LEARNING_RATE - the learning rate to use
# MOMENTUM - the momentum to use
# BETA - the beta to use
# BETA1 - the beta1 to use
# BETA2 - the beta2 to use
# EPSILON - the epsilon to use
# WEIGHT_DECAY - the weight decay to use
# WEIGHT_INIT - the weight initialization to use
# NUM_LAYERS - the number of layers to use
# HIDDEN_SIZE - the size of the hidden layers
# ACTIVATION - the activation function to use

import Files.settings as settings

def set_paramaters(arguments_array):
    # Description: This function sets the parameters from the arguments array
    # Arguments: arguments_array - array of arguments
    # Returns: None

    # arguments_array[0] is the name of the file
    # others are pairs of arguments
    for i in range(1, len(arguments_array), 2):

        if arguments_array[i] == "-wp" or arguments_array[i] == "--wandb_project":
            settings.WANDB_PROJECT = arguments_array[i+1]

        elif arguments_array[i] == "-we" or arguments_array[i] == "--wandb_entity":
            settings.WANDB_ENTITY = arguments_array[i+1]

        elif arguments_array[i] == "-d" or arguments_array[i] == "--dataset":
            settings.DATASET = arguments_array[i+1]

        elif arguments_array[i] == "-e" or arguments_array[i] == "--epochs":
            settings.EPOCHS = int(arguments_array[i+1])

        elif arguments_array[i] == "-b" or arguments_array[i] == "--batch_size":
            settings.BATCH_SIZE = int(arguments_array[i+1])

        elif arguments_array[i] == "-l" or arguments_array[i] == "--loss":
            settings.LOSS = arguments_array[i+1]

        elif arguments_array[i] == "-o" or arguments_array[i] == "--optimizer":
            settings.OPTIMIZER = arguments_array[i+1]

        elif arguments_array[i] == "-lr" or arguments_array[i] == "--learning_rate":
            settings.LEARNING_RATE = float(arguments_array[i+1])

        elif arguments_array[i] == "-m" or arguments_array[i] == "--momentum":
            settings.MOMENTUM = float(arguments_array[i+1])

        elif arguments_array[i] == "-beta" or arguments_array[i] == "--beta":
            settings.BETA = float(arguments_array[i+1])

        elif arguments_array[i] == "-beta1" or arguments_array[i] == "--beta1":
            settings.BETA1 = float(arguments_array[i+1])

        elif arguments_array[i] == "-beta2" or arguments_array[i] == "--beta2":
            settings.BETA2 = float(arguments_array[i+1])

        elif arguments_array[i] == "-eps" or arguments_array[i] == "--epsilon":
            settings.EPSILON = float(arguments_array[i+1])

        elif arguments_array[i] == "-w_d" or arguments_array[i] == "--weight_decay":
            settings.WEIGHT_DECAY = float(arguments_array[i+1])

        elif arguments_array[i] == "-w_i" or arguments_array[i] == "--weight_init":
            settings.WEIGHT_INIT = arguments_array[i+1]

        elif arguments_array[i] == "-nhl" or arguments_array[i] == "--num_layers":
            settings.NUM_LAYERS = int(arguments_array[i+1])

        elif arguments_array[i] == "-sz" or arguments_array[i] == "--hidden_size":
            settings.HIDDEN_SIZE = int(arguments_array[i+1])

        elif arguments_array[i] == "-a" or arguments_array[i] == "--activation":
            settings.ACTIVATION = arguments_array[i+1]

        else:
            print("Error: Invalid argument: " + arguments_array[i])