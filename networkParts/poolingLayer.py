from networkParts.layer import Layer
import numpy as np

class PoolingLayer1D(Layer):
    def __init__(self, input_size, kernel_size=3, step_size=1, mode="max"):
        self.input_size=input_size
        self.output_size=(input_size-kernel_size)+1
        self.kernel_size=kernel_size
        self.step_size=step_size
        self.mode=mode

        #Make sure the mode is set properly
        if mode != "max" and mode != "avg":
            raise Exception("Mode must either be set to \"max\" or \"avg\"")
        
    def forward_propagation(self, input_data):
        #setup
        self.input=input_data
        mode=self.mode
        output_data=[]
        
        for i in range(0,self.input_size-self.kernel_size+1,self.step_size):
            current_kernel=self.input[i:i+self.kernel_size] #grab the current kernel
            if mode=="max":
                output_data.append(np.max(current_kernel))
            if mode=="avg":
                output_data.append(np.average(current_kernel))
        return output_data

    def backward_propagation(self, output_error, learning_rate):
        #Nothing should be done here. This part of the network doesn't have weights.
        input_error=output_error
        return input_error
