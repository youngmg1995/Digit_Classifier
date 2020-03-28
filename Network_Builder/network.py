# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:56:24 2020

@author: Mitchell
"""

"""network.py
~~~~~~~~~~~~~~
This file contains a set of classes and modules that can be used for building
and training fully connected feedforward neural networks. Each network is
trained using the stochastic gradient descent learning algorithm with
backpropogation. Additionally, each network can be saved and loaded using a
json format for later use.

No advanced machine learning modules are leveraged for thiscode, and the only
non-standard library required in numpy. 

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


#### Define Cost class and each of our individual cost functions. Namely the 
#### quadratic and cross-entropy cost functions.
class Cost(object):
    pass

class QuadraticCost(Cost):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def der(a, y):
        """Return the gradient of the cost function wrt the activations."""
        return a-y


class CrossEntropyCost(Cost):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1.0-y)*np.log(1.0-a)))

    @staticmethod
    def der(a, y):
        """Return the gradient of the cost function wrt the activations.
        Note that np.nan_to_num is used to ensure numerical stability."""
        return np.nan_to_num((a-y) / (a*(1.0-a)))

#### Define the Activation class and each of the possible activation functions.
class Activation(object):
    pass
    
class Perceptron(Activation):
    
    @staticmethod
    def fn(z):
        """Returns the value of our activation function for the given
        pre-activation.
        """
        return np.heaviside(z, 0.)

    @staticmethod
    def der(z):
        """Returns the derivative of our activation function for the given
        pre-activation.
        """
        return 0.*z
    
class Linear(Activation):
    
    @staticmethod
    def fn(z):
        """Returns the value of our activation function for the given
        pre-activation.
        """
        return z

    @staticmethod
    def der(z):
        """Returns the derivative of our activation function for the given
        pre-activation.
        """
        return 1.0 + 0.*z

class Sigmoid(Activation):
    
    @staticmethod
    def fn(z):
        """Returns the value of our activation function for the given
        pre-activation.
        """
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def der(z):
        """Returns the derivative of our activation function for the given
        pre-activation.
        """
        return np.exp(-z) / (1.0+np.exp(-z))**2
    
class TanH(Activation):
    
    @staticmethod
    def fn(z):
        """Returns the value of our activation function for the given
        pre-activation.
        """
        return 2.0/(1.0+np.exp(-2.0*z)) - 1.0

    @staticmethod
    def der(z):
        """Returns the derivative of our activation function for the given
        pre-activation.
        """
        return 1.0 - (2.0/(1.0+np.exp(-2.0*z)) - 1.0)**2
    
class ArcTan(Activation):
    
    @staticmethod
    def fn(z):
        """Returns the value of our activation function for the given
        pre-activation.
        """
        return np.arctan(z)

    @staticmethod
    def der(z):
        """Returns the derivative of our activation function for the given
        pre-activation.
        """
        return 1.0/(z**2 + 1.0)
    
class ReLU(Activation):
    
    @staticmethod
    def fn(z):
        """Returns the value of our activation function for the given
        pre-activation.
        """
        return np.maximum(0., z)

    @staticmethod
    def der(z):
        """Returns the derivative of our activation function for the given
        pre-activation.
        """
        return np.heaviside(z, 0.)
    
class PReLU(Activation):
    
    def __init__(self, a):
        """Initializes the one parameter of the activation. Namely, stores the
        prescribed slope for our linear unit.
        """
        self.a = a
    
    def fn(self, z):
        """Returns the value of our activation function for the given
        pre-activation.
        """
        return np.maximum(0., self.a*z)

    def der(self, z):
        """Returns the derivative of our activation function for the given
        pre-activation.
        """
        return np.heaviside(z, 0.)*self.a
    
class SoftPlus(Activation):
    
    @staticmethod
    def fn(z):
        """Returns the value of our activation function for the given
        pre-activation.
        """
        return np.log(1.0+np.exp(z))

    @staticmethod
    def der(z):
        """Returns the derivative of our activation function for the given
        pre-activation.
        """
        return 1.0/(1.0+np.exp(-z))
    
class SoftMax(Activation):
    
    @staticmethod
    def fn(z):
        """Returns the value of our activation function for the given
        pre-activation. Note to ensure stability we subtract maximum z value
        from our z-array.
        """
        z_max = np.max(z)
        return np.exp(z-z_max)/np.sum(np.exp(z-z_max))

    @staticmethod
    def der(z):
        """Returns the derivative of our activation function for the given
        pre-activation. Note to ensure stability we subtract maximum z value
        from our z-array.
        """
        z_max = np.max(z)
        return np.exp(z-z_max)/np.sum(np.exp(z-z_max)) - (np.exp(z-z_max)/np.sum(np.exp(z-z_max)))**2
        
#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost, activations = None):
        """Initializes a network instance using the prescribed network sizes,
        cost function, and activations. The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method). Also initializes the weight and bias velocity arrays that will
        be used during SGD training with 0's.
        
        sizes:  List containing the number of nodes in each layer of our.
                network. Example - sizes = [784, 100, 100, 10].
        
        cost:   Cost function used to evaluate training examples. Should be one
                of the functions defined above from the Cost class. By default
                is set to the CrossEntropyCost function as defined above.
                Example - cost = CrossEntropyCost
                
        activations:    List of the activation functions of our network.
                Specifically, the activation functions that will be applied to
                each non-input layer of our network. By default will make the
                hidden layer activation function ReLU and the output layer
                SoftMax.
                Example - activations = [Sigmoid, Sigmoid, SoftMax]
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.biases_velocity = [np.zeros([y, 1]) for y in self.sizes[1:]]
        self.weights_velocity = [np.zeros([y, x])
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.cost=cost
        if activations:
            self.activations = activations
        else:
            self.activations = []
            for i in range(len(self.biases)):
                if i == len(self.biases) - 1:
                    self.activations.append(SoftMax)
                else:
                    self.activations.append(ReLU)
        self.evaluations = 0

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1. Also initializes

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def large_weight_initializer(self):
        """Initialize each weight and bias using a Gaussian distribution with
        mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def constant_weight_initializer(self, constant):
        """Initialize the weights and biases to 0 if desired. May be beneficial
        for certain activations and cost functions

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.ones([y, 1])*constant for y in self.sizes[1:]]
        self.weights = [np.ones([y, x])*constant
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def velocity_reset(self):
        """Sets gradient descent velocity for weights and biases to 0. This is
        a helper function used during SGD training to reset weight and bias
        velocities. Currently only being used to reset velocities before each
        new SGD run, but can be implemented elsewhere if desired (Example: 
        after each training epoch reset velocities)."""
        self.biases_velocity = [np.zeros([y, 1]) for y in self.sizes[1:]]
        self.weights_velocity = [np.zeros([y, x])
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w, activation in zip(self.biases, self.weights, self.activations):
            a = activation.fn(np.dot(w, a)+b)
        self.evaluations += 1
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            max_norm = None,
            max_norm_ratio = 1.,
            dropout = 0.,
            input_dropout = 0.,
            eta_decay = 1.,
            momentum = False,
            friction = 1.,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n = 0):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, but are also 
        described below according to their associated technique. The method 
        also accepts ``evaluation_data``, usually either the validation or
        test data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        Added Functionalities:
        1) Gradient Descent Modifications:
        Stochastic gradient descent training algorithm has been modified to allow
        eta decay over the cours of epochs and to allow the use of momentum.
        
        2) Dropout Backpropogation:
        Backpropogation algorithm has been modified to allow dropout implementation
        during training. This allows for a specified fraction of the inputs and
        hidden nodes to be ignored during the training of each mini-batch.
        
        3) Generalized Cost Functions:
        Have generalized the cost functions into their own class and changed
        function to have two functions, evaluation and derivative. These calculate
        the value of the cost function and its derivative respectively for the
        given activations and solutions. (This was done to allow the use of
        activation functions other than the sigmoid function.)
        
        4) Generalized Activations:
        Like the cost functions, the activations have been generalized into their
        own class with two methods, evaluation and its derivative. Additionally,
        more activation functions have been added and the user now has ability to
        specify a different activation function for each layer of the network. 
        (This was done primarily to support the use of a softmax output activation 
        so the results are a probability distribution. However, decided that it
        wouldn't take much more work to allow each layer to have an independent 
        activation prescription, allowing much more freedom and customizability
        to each network.)
        
        5) Regularization Techniques:
        In addition to the dropout regularization mentioned above, there are two
        types of weight contraint regularization techniques that can be used in
        our new networks. The first is using L2 weights decay regularization and
        the second using max-norm weight constraints. The former adds a term to our
        cost function that is a sum of the squares of all the weights in our
        network. This encourages our network weights to shrink over time through
        the gradient descent updates. The latter enforces a limit on the frobenius
        norm of the weights at hidden nodes. In other words, it constrains the norm
        of each node's weights to be below a certain value. This prevents the
        wieghts from getting large by restricting them to a small subspace, but
        does not force them to get smaller over time.
        """

        # early stopping functionality:
        best_accuracy=1

        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy=0
        no_accuracy_change=0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        self.velocity_reset()
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for i in range(len(self.biases)):
                M=self.sizes[i]
                if i == 0:
                    m = int(M - input_dropout*M)
                else:
                    m = int(M - dropout*M)
                self.weights[i] = self.weights[i] * M/m
            
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta*(eta_decay**j), lmbda, dropout, input_dropout,
                    friction)
                if max_norm:
                    self.normalize_weights(max_norm, max_norm_ratio)

            for i in range(len(self.biases)):
                M=self.sizes[i]
                if i == 0:
                    m = int(M - input_dropout*M)
                else:
                    m = int(M - dropout*M)
                self.weights[i] = self.weights[i] * m/M

            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    #print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    #print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, dropout, input_dropout, friction):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
   
        node_arrays = []
        for i in range(len(self.biases)):
            N=self.sizes[i]
            if i == 0:
                n = int(N - input_dropout*N)
            else:
                n = int(N - dropout*N)
            node_arrays.append(sorted(np.random.choice(range(N),n, replace=False)))
        node_arrays.append(list(range(self.sizes[-1])))
        
        node_matrices = []
        biases = []
        weights = []
        biases_velocity = []
        weights_velocity = []
        for i in range(len(self.biases)):
            node_matrix = np.meshgrid(node_arrays[i], node_arrays[i+1])
            node_matrix.reverse()
            node_matrices.append(node_matrix)
            nodes = node_arrays[i+1]
            biases.append(self.biases[i][nodes])
            biases_velocity.append(self.biases_velocity[i][nodes])
            N=self.sizes[i]
            if i == 0:
                n = int(N - input_dropout*N)
            else:
                n = int(N - dropout*N)
            weights.append(self.weights[i][tuple(node_matrix)])
            weights_velocity.append(self.weights_velocity[i][tuple(node_matrix)])
        
        
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        for x, y in mini_batch:
            x = x[node_arrays[0]]
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, biases, weights)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        weights_velocity = [(1-friction)*wv-(eta/len(mini_batch))*nw
                            for wv, nw in zip(weights_velocity, nabla_w)]
        biases_velocity = [(1-friction)*wb-(eta/len(mini_batch))*nb
                           for wb, nb in zip(biases_velocity, nabla_b)] 
        weights = [(1-eta*lmbda)*w + wv
                   for w, wv in zip(weights, weights_velocity)]
        biases = [b-(eta/len(mini_batch))*bv
                  for b, bv in zip(biases, biases_velocity)]
        
        for i in range(len(self.biases)):
            nodes = node_arrays[i+1]
            self.biases[i][nodes] = biases[i]
            self.biases_velocity[i][nodes] = biases_velocity[i]
            node_matrix = node_matrices[i]
            N = self.sizes[i]
            if i == 0:
                n = int(N - input_dropout*N)
            else:
                n = int(N - dropout*N)
            self.weights[i][tuple(node_matrix)] = weights[i]
            self.weights_velocity[i][tuple(node_matrix)] = weights_velocity[i]
            
        

    def backprop(self, x, y, biases, weights):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w, act in zip(biases, weights, self.activations):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = act.fn(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).der(activations[-1], y) * (self.activations[-1]).der(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = (self.activations[-l]).der(z)
            delta = np.dot(weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def normalize_weights(self, max_norm, max_norm_ratio):
        """Re-normalizes the weights of each hidden node in our network to have
        a frobenius norm less than the prescribed max-norm value. This function
        basically enforces our max-norm regularization constraint during SGD if
        prescribed."""
        for i in range(len(self.weights)-1):
            norms = np.linalg.norm(self.weights[i], axis = 1)
            test = norms > max_norm
            if len(norms[test]) > 0:
                norms = norms.reshape((len(norms),1))
                self.weights[i][test] = self.weights[i][test] * max_norm / norms[test] / max_norm_ratio
            

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy
    
    def inaccuracy(self, data, convert=False):
        """Return the inputs in ``data`` for which the neural
        network outputs the correct result.
        
        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data
        """
        
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        
        incorrect_images = []
        incorrect_digits = []
        correct_digits = []
        image_indices = []
        for i in range(len(results)):
            if results[i][0] != results[i][1]:
                incorrect_images.append(data[i][0])
                incorrect_digits.append(results[i][0])
                correct_digits.append(results[i][1])
                image_indices.append(i)
        return incorrect_images, incorrect_digits, correct_digits, image_indices

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda)*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "activations":[str(activation.__name__) for activation in self.activations],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    activations = [getattr(sys.modules[__name__], activation) for activation in data["activations"]]
    net = Network(data["sizes"], cost=cost, activations = activations)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e