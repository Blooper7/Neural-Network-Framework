class Network:
    def __init__(self):
        self.layers=[]
        self.loss=None #Loss func to use
        self.loss_prime=None #Loss func derivative

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss=loss
        self.loss_prime=loss_prime

    def predict(self, input_data, allData=False):
        samples=len(input_data) #Grab the dimension count
        result=[]

        for i in range(samples):
            output=input_data[i]
            for layer in self.layers:
                output=layer.forward_propagation(output)
                #print(output)
            result.append(output)
        
        
        if allData:
            return result
        return result[-1]

    def train(self, x_train, y_train, epochs, learning_rate, verbose=False):
        #For future me: an epoch is one pass of all training data
        #x_train[i] is the input, y_train[i] is the expected output
        samples=len(x_train)
        error_amts=[]

        #training loop
        for i in range(epochs):
            err=0
            for j in range(samples):
                #forward propagation
                output=x_train[j]
                for layer in self.layers:
                    output=layer.forward_propagation(output)
                
                #compute loss (mainly for display)
                err += self.loss(y_train[j], output)

                #backward propagation
                error=self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error=layer.backward_propagation(error, learning_rate)

            #calculate the average error
            err/=samples

            error_amts.append(err)

            if verbose:
                print('epoch %d/%d   error=%f' % (i+1, epochs, err))
        return error_amts