'''
This file contains simple functions to test with.
'''
import numpy as np


def f1(x, mean=0.0, std=1.0):
    '''
    '''
    x = np.array(x)
    return (1.0 / np.sqrt(2.0 * np.pi * std**2.0)) *\
        np.exp(- (x - mean)**2.0 / (2.0 * std**2))


def f2(x):
    '''
    Product of a periodic and gaussian function.
    '''
    return np.sin(x)**2 * np.exp((2 - x)**2 / 10.0)
