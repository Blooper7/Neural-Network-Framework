#from layer import Layer
from networkParts.layer import Layer
import numpy as np

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1,output_size) - 0.5
        self.input_size=input_size
        self.output_size=output_size

    def forward_propagation(self, input_data:list[list[int]]):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

if __name__ == "__main__":
    layer=FCLayer(5, 2)
    print(layer.forward_propagation([[0,1,2,3,4]]))