from networkParts.layer import Layer
import numpy as np

class ActivationFunctions:
    #TODO: implement sigmoid
    #Tanh function
    def tanh(x,c=1): #C is a coefficient for putting the results in a range. EX: a C of 10 means the result (0 to 1) will be 0-10
        return c*np.tanh(x)
    def tanh_prime(x,c=1):
        return c*(1-np.tanh(x)**2)
    
    #Sigmoid function
    def sigmoid(x,c=1):
        return c*(1/(1+np.exp(-x)))
    def sigmoid_prime(x,c=1):
        return c*( (c*(1/(1+np.exp(-x)))) * (1- (c*(1/(1+np.exp(-x)))) ))

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime, coefficient=1):
        self.activation=activation
        self.activation_prime=activation_prime
        self.coefficient=coefficient

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input, self.coefficient)
        return self.output
    
    #Returns input error for a given output error
    #Learning rate unused due to the lack of learnable parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error