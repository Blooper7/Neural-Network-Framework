from layer import Layer
import numpy as np

def calc_valid_convolution_size(size_input, size_kernel, steps):
    val=size_input
    for i in range(steps):
        if val<size_kernel:
            raise Exception("The kernel size exceeded the input size. Ensure the kernel stays smaller than the input size.")
        val=(val-size_kernel)+1
    return val

class ConvolutionLayer1D(Layer):
    def __init__(self, input_size, kernel_size=3, step_size=1, mode="valid"):
        self.input_size=input_size
        self.output_size=(input_size-kernel_size)+1
        self.weights=np.random.rand(kernel_size)-0.5
        self.bias=np.random.rand(self.output_size) - 0.5
        self.step_size=step_size
        self.mode=mode

        if self.input_size%self.step_size!=0:
            raise Exception("Input size must be fully divisible by the step size, leaving no remainder.")
        if len(self.weights)>self.input_size:
            raise Exception("The kernel has to fit within the input!")
    
    def forward_propagation(self, input_data):
        self.input=input_data
        '''while len(input_data)<self.input_size:
            self.input[0].append(0)'''
        print(f"{self.input=}")
        print(f"{self.weights=}")
        output_data=np.convolve(self.input, self.weights, mode=self.mode)
        #print(f"{output_data=}")
        #print(f"{self.bias=}")
        self.output=output_data+self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        input_error=np.convolve(output_error, self.weights[::-1], mode='full')
        weights_error=np.convolve(self.input, output_error, mode='valid')
        self.weights-=learning_rate*weights_error
        self.bias-=learning_rate*np.sum(output_error)
        return input_error


    def __str__(self):
        return f"Weights {self.weights}, Bias {self.bias}"


if __name__ == "__main__":
    myLayer=ConvolutionLayer1D(6, 3, 1, "valid")
    myLayer.weights=np.array([0.5, -0.5, 1])
    myLayer2=ConvolutionLayer1D(4, 2, 1, "valid")
    myLayer2.weights=np.array([0.5, -0.5])
    #print(myLayer.bias)
    print(myLayer2.forward_propagation(myLayer.forward_propagation(np.array([1,2,3,4,5,6]))))