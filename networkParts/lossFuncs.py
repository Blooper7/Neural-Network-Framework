import numpy as np

class LossFunctions:
    #Mean Squared Error, the error between the true output and the predicted output
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))

    def mse_prime(y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.size
