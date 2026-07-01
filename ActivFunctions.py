import numpy as np
#this is .py file with activation functions and their derivatives


#identity
def identity(x):
    return x

def der_identity(x):
    return 1
#Sigmoid
def sigmoid(x):
    # numerically stable
    return 1.0 / (1.0 + np.exp(-x))

def der_sigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)
#Tanh
def tanh(x):
    return np.tanh(x)

def der_tanh(x):
    t = np.tanh(x)
    return 1.0 - t**2
#ReLU
def relu(x):
    return np.maximum(0.0, x)

def der_relu(x):
    return 1.0 * (x > 0.0)
# Leaky ReLU

def leaky_relu(x, slope=0.01):
    return np.where(x > 0, x, slope * x)

def der_leaky_relu(x, slope=0.01):
    return np.where(x > 0, 1.0, slope)
#Softmax (scalar input) - if softmax is used in scalar context, then it is dividing by itself, so it would yield 1 and derivative would be always 0. In practice it should never happen, but it is up to user choice
def softmax(x):
    return 1

def der_softmax(x):
    return 0

#Softmax (vector/array input)
def softmax_vec(x):
    #getting maximal value in each example (column) to ensure numerical stability in next step
    x_max_val = np.max(x, axis=0, keepdims=True)#keepdims=True keeps shape for broadcasting(which should be impossible due to failsafes in previous steps)
    #maximal (largest) value is substracted to avoid numerical overflow
    x_shifted = x - x_max_val
    #values are taken into exponential to get probabilities (yet to be valid probabilities)
    exp_values = np.exp(x_shifted)
    #getting sum of all values in each example so valid probabilities (in range of 0-1) can be obtained
    x_sum = np.sum(exp_values, axis=0, keepdims=True)
    #calculating valid class probabilities in each example
    output = exp_values / x_sum
    #results are returned
    return output
def der_softmax_vec(x):
    #as in this implementation activation function derivatives are calculated from z (pre-activation results) and in this case they are needed for derivative calculation
    activation_results = softmax_vec(x)
    #getting shape of activation results is needed for 
    activation_results_shape = activation_results.shape
    num_class = activation_results_shape[0]
    batch_size = activation_results_shape[1]
    #output variable is declared for pre-allocation. In case of softmax function, the derivative is jacobian matrix
    jacobian = np.zeros((batch_size,num_class,num_class))
    #iterating through all examples as each needs its own jacobian matrix
    for iExample in range(batch_size):
        #getting activation results for current example -> jacobian is calculate y example
        activation_result = activation_results[:,iExample].reshape(-1,1)
        #calculating jacobian matrix for current example
        jac_mat = np.diagflat(activation_result) - activation_result @ activation_result.T
        #assigning results to output variable
        jacobian[iExample] = jac_mat
    #returning output
    return jacobian
