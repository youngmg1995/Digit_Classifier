# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:18:11 2020

@author: Mitchell
~~~~~~~~~~~~~~~~~

This version of the network builder uses tensorflow to build and train our 
nueral networks using GPUs. Code below was tested using tensorflow=2.0.0. For
a good tutorial on how to create a working python environment with tensorflow
use the below link:
    
    https://www.youtube.com/watch?v=qrkEYf-YDyI
    
Using tensorflow lets us more easily build neural networks using the pre-built
and optimized methods inherent to tensorflow, rather than trying to build our
own. Additionally, running these networks and the general code on a GPU 
significantly improves speed and performance. Networks that took days to build 
using "Network_Builder.py" run in tensorflow-gpu in hours. For information on
tensorflow and help understanding the implementation and methods below see the
documentation page listed below:
    
    https://www.tensorflow.org/
    
Note: Networks 1, 2, and 3 built in this script have the same structures,
parameters, and training prescriptions as the corresponding networks in
"Network_Builder". The purpos of this was to compare the ease of constructing
the same networks using tensorflow and to compare the performance and training
speed, expecially in the case of the convolutional neural network. Because this
was our goal, the tensorflow networks seen here are not optimized. There are
several parameters and objects that were overridden from their defaults and
best practices that were diregarded in order to recreate the more basic
networks contructed using numpy. For instance, here we continue to choose our
initial weights and biases froma gaussian ditributions despite that in practice 
this has shown to be innefective for DNNs and is not the default initializer
for tensorflow layers.
"""

# Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os


### Loading and Preparing MNIST Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loading in the MNIST images dataset which is already within the tensorflow 
# library
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Transforming images formats that can be used in each of our networks
x_train , x_test = x_train.astype('float32')/255.0 , x_test.astype('float32')/255.0
alt_x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
alt_x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
alt_y_train = tf.keras.utils.to_categorical(y_train, 10)
alt_y_test = tf.keras.utils.to_categorical(y_test, 10)



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

# define weight/bias initializer
initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)
# define structure of our first neural network
net_1 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(30, activation='sigmoid',
                        kernel_initializer = initializer,
                        bias_initializer = initializer),
  tf.keras.layers.Dense(10, activation='sigmoid',
                        kernel_initializer = initializer,
                        bias_initializer = initializer)
])

# define loss function for training
loss_fn = tf.keras.losses.MeanSquaredError()
# define update algorithm for training
grad_desc = tf.keras.optimizers.SGD(learning_rate=3.0)

# configure our neural network for training
net_1.compile(optimizer = grad_desc, loss = loss_fn,
              metrics = ['accuracy']
)
# train our nueral network
net_1.fit(x_train, alt_y_train, batch_size = 10, epochs = 30,
          validation_data = (x_test, alt_y_test))

# save model for use later
foldername = 'Saved_Networks\\'
filename = 'network1_tf'
path = os.path.join(foldername, filename)
if not os.path.exists(path):
    os.mkdir(path) 
net_1.save(foldername+filename, save_format='tf')



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

# define structure of our first neural network
net_2 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(800, activation='relu',
                        kernel_initializer = tf.keras.initializers.lecun_normal(),
                        bias_initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0),
                        kernel_constraint = tf.keras.constraints.MaxNorm(max_value=4)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(800, activation='relu',
                        kernel_initializer = tf.keras.initializers.lecun_normal(),
                        bias_initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0),
                        kernel_constraint = tf.keras.constraints.MaxNorm(max_value=4)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(10, activation='softmax',
                        kernel_initializer = tf.keras.initializers.lecun_normal(),
                        bias_initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0))
])

# define loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
# define the exponential decay of our learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    .1,
    decay_steps=60000,
    decay_rate=0.995,
    staircase=True)
# define update algorithm for training
grad_desc = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.5)

# configure our neural network for training
net_2.compile(optimizer = grad_desc, loss = loss_fn,
              metrics = ['accuracy']
)
# train our nueral network
net_2.fit(x_train, y_train, batch_size = 100, epochs = 1000,
          validation_data = (x_test, y_test))

# save model for use later
foldername = 'Saved_Networks\\'
filename = 'network2_tf'
path = os.path.join(foldername, filename)
if not os.path.exists(path):
    os.mkdir(path) 
net_2.save(foldername+filename, save_format='tf')



### Network 3:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Network: 
#   -Structure: Convolutional Neural Network
#       -Layer 1: Convolutional Pooling Layer
#           -Input: 1 x 28 x 28 image
#           -#  of Feature Maps: 20
#           -Local Receptive Field: 5 x 5
#           -Stride Length: 1
#           -Pooling: 2 x 2 input region Max Pooling
#       -Layer 2: Convolutional Pooling LAyer
#           -Input: 20 x 12 x 12 output of Convo. Layer 1
#           -#  of Feature Maps: 20
#           -Local Receptive Field: 5 x 5
#           -Stride Length: 1
#           -Pooling: 2 x 2 input region Max Pooling
#       -Layer 3: Dense Neural Network
#           -Structure: [640,100,10]
#           -Activations: ReLU for 100 hidden nodes, SoftMax for Output Layer
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

# define structure of our convolutional neural netwrork
net_3 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(20, (5,5), activation = 'relu', input_shape = (28,28,1)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(40, (5,5), activation = 'relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(100, activation='relu',
                        kernel_initializer = tf.keras.initializers.lecun_normal(),
                        bias_initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)),
  tf.keras.layers.Dense(10, activation='softmax',
                        kernel_initializer = tf.keras.initializers.lecun_normal(),
                        bias_initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0))
])

# define loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
# define the exponential decay of our learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    .01,
    decay_steps=60000,
    decay_rate=0.995,
    staircase=True)
# define update algorithm for training
grad_desc = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.5)

# configure our neural network for training
net_3.compile(optimizer = grad_desc, loss = loss_fn,
              metrics = ['accuracy']
)
# train our nueral network
net_3.fit(alt_x_train, y_train, batch_size = 100, epochs = 1000,
          validation_data = (alt_x_test, y_test))

# save model for use later
foldername = 'Saved_Networks\\'
filename = 'network3_tf'
path = os.path.join(foldername, filename)
if not os.path.exists(path):
    os.mkdir(path) 
net_3.save(foldername+filename, save_format='tf')



