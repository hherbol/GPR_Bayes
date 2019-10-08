'''
This file holds the Theta object, which will robustly handle
hyperparameters.
'''
import numpy as np


class Theta:
    '''
    '''

    def __init__(self, kernel, domain_shape):
        kernel = kernel.lower()
        self.kernel = kernel

        self.error_handle()

        if len(domain_shape) == 1:
            dim = 1
        else:
            dim = domain_shape[1]

        if kernel == "mk52":
            # The number of parameters is equal to the dimensionality,
            # to account for the weights, and one extra for the standard
            # deviation
            self.size = dim + 1
        elif kernel == "pk":
            self.size = dim + 2
        elif kernel == "other":
            self.size = None
        else:
            raise Exception("Invalid kernel in Theta.")

    def error_handle(self):
        assert self.kernel in [
            "mk52",
            "pk",
            "other"
        ], "Error - Kernel (%s) has not been setup with Theta yet." % self.kernel

    def sample(self, nsample):
        '''
        Sample randomly the kernel hyperparameter space.
        '''
        self.error_handle()

        if self.kernel == "mk52":
            # Weights can be between 0.01 and 1.0, while
            # standard deviation is set between 0.001 and 10.0
            bounds = np.array([
                (0.01, 1.0) for i in range(self.size - 1)
            ] + [(0.001, 10.0)])
            samples = np.random.random((nsample, self.size))
            bound_magnitudes = bounds[:, 1] - bounds[:, 0]
            samples = (samples - bounds[:, 0]) * bound_magnitudes
            return samples, bounds
        elif self.kernel == "pk":
            # Weights can be between 0.01 and 1.0, while
            # standard deviation is set between 0.001 and 10.0
            # and period is set between 0 and 2pi
            bounds = np.array([
                (0.01, 1.0) for i in range(self.size - 2)
            ] + [(0.001, 10.0)] + [(0.001, 100.0)])
            samples = np.random.random((nsample, self.size))
            bound_magnitudes = bounds[:, 1] - bounds[:, 0]
            samples = (samples - bounds[:, 0]) * bound_magnitudes
            return samples, bounds
        elif self.kernel == "other":
            raise Exception("Hasn't been handled yet")
        else:
            raise Exception("Invalid kernel in Theta.")

    def update(self, values):
        '''
        Store hyperparameters
        '''
        self.values = np.array(values)
