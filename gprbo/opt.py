'''
This file contains a simple GPR Bayesian Optimizer
to predict a simple N-dim function.
'''
from 


class GPR_BO:
    '''
    '''
    def __init__(self):
        pass


if __name__ == "__main__":
    opt = GPR_BO()
    # Set a zero prior mean
    opt.set_prior_mean(lambda x: 0.0)
    # Set the covariance function
    opt.set_cov_func()
