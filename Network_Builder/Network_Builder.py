# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 23:35:26 2020

@author: Mitchell
"""
### Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import mnist_loader, network, time


### Loading and Preparing MNIST Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)


### Network 1: 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Network: 
#   -Structure: Single hiddden layer (784,30,10)
#   -Activations: Sigmoid neurons for hidden and output layer
#   -Initialization: Random starting weights/biases chosen from gaussina distr.
#       -Weigths and Biases: mean = 0, std = 1
#
# Training:
#   -Quadratic Cost Function
#   -Stochastic Gradient Descent
#       -Learning Speed: Eta = 3.0 
#   -Trained for 30 epochs with minibatches of size 10
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# define our network and inialize
net_1 = network.Network([784, 30, 10], cost=network.QuadraticCost,
                      activations = [network.Sigmoid, network.Sigmoid])
net_1.large_weight_initializer()

# train our neural network
tic = time.perf_counter()
net_1.SGD(training_data, 30, 10, 3.0, evaluation_data=test_data,
        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
        monitor_training_cost=True, monitor_training_accuracy=True)
toc = time.perf_counter()
print(f"Created network1 in {toc - tic:0.4f} seconds")

# save network to a json file for later use
foldername = 'Saved_Networks/'
filename = 'network1.json'
net_1.save(foldername+filename)
print('Network1 saved to file \''+foldername+filename+'\'')



### Network 2:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Network: 
#   -Structure: Two hiddden layers (784,800,800,10)
#   -Activations: ReLU hidden layers and SoftMax output layer
#   -Initialization: Random starting weights/biases chosen from gaussina distr.
#       -Biases: mean = 0, std = 1
#       -Weights: mean = 0, std = 1/(# inputs)^(1/2)
#
# Training:
#   -Cross Entropy Cost Function
#   -Regularization 
#       -max-norm wieght contraint with l=4
#       -Dropout: Hidden Layers Rate = .5, Input Layer Rate = .2
#   -Momentum Based Stochastic Gradient Descent
#       -Friction Coeff = .5
#       -Learning Speed: Eta = .5*.995^(t) where t is the epoch number 
#   -Trained for 1000 epochs with minibatches of size 100
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# define our network and inialize
net_2 = network.Network([784, 800, 800, 10])
tic = time.perf_counter()

# train our neural network
net_2.SGD(training_data, 1000, 100, .5, evaluation_data=test_data,
        eta_decay = .995,
        friction = .5,
        dropout = .5,
        input_dropout = .2,
        max_norm = 4,
        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
        monitor_training_cost=True, monitor_training_accuracy=True)
toc = time.perf_counter()
print(f"Created network1 in {toc - tic:0.4f} seconds")

# save network to a json file for later use
foldername = 'Saved_Networks/'
filename = 'network2.json'
net_2.save(foldername+filename)
print('Network2 saved to file \''+foldername+filename+'\'')



