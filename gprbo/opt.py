'''
This file contains a simple GPR Bayesian Optimizer
to predict a simple N-dim function.
'''
from gprbo.kernels.matern import maternKernel52 as mk52
from gprbo.hyperparameters import Theta
from gprbo.testFunctions.simple import f1

import copy
import numpy as np
import scipy.linalg
from scipy import optimize as op


class GPR_BO:
    '''
    '''
    def __init__(self, domain):
        self.domain = np.array(domain)
        if len(self.domain.shape) == 1:
            self.domain = self.domain
        self.mean = lambda x: 0.0
        self.cov = mk52
        self.HP = Theta("mk52", self.domain.shape)

    def likelihood(self, X, Y, HPs):
        '''
        This function computes the likehood of solubilities given
        hyperparameters in the list theta.

        Note - when doing miso, X is a reduced list (say we sample
        10 at each of 2 IS in the beginning, X is then 10 long, but
        Y is 20 long).

        **Parameters**

            X:
                A list of the sampled X coordinates.
            Y:
                A list of the objectives calculated, corresponding
                to the different X values.
            length:
                The one-d numpy array of keys for the hps.

        **Returns**

            likelihood: *float*
                The log of the likelihood without the constant term.
        '''
        # length, sigma = blarg
        X = np.array([np.array(x) for x in X])

        mu = map(self.mean, X)
        Sig = self.cov(X, X, HPs)

        Sig = Sig + np.eye(len(Sig)) * 1E-6

        y = Y - mu
        L = scipy.linalg.cho_factor(Sig, lower=True, overwrite_a=False)
        alpha = scipy.linalg.cho_solve(L, y)

        val = -0.5 * y.T.dot(alpha)
        val -= sum([np.log(x) for x in np.diag(L[0])])
        val -= len(mu) / 2.0 * np.log(2.0 * np.pi)

        return val

    def fit_hp(self, tx, ty):
        '''
        Given training data, find hyperparameters that maximize the MLE.
        '''
        f = lambda *args: -1.0 * self.likelihood(tx, ty, *args)

        # Get initial values
        init_values, bounds = self.HP.sample(5)
        mle_list = np.zeros([len(init_values), self.HP.size])

        lkh_list = np.zeros(len(init_values))
        for i, v in enumerate(init_values):
            results = op.minimize(f, v, bounds=bounds)
            mle_list[i, :] = results['x']  # Store the optimized parameters
            lkh_list[i] = results.fun  # Store the resulting likelihood

        # print lkh_list
        self.HP.update(mle_list[np.nanargmax(lkh_list), :])

    def train(self, tx, ty, use_least_sq=True):
        '''
        This function will train the domain against the
        sampled training points.
        '''
        assert self.HP is not None, "Error - first set hyperparameters!"

        mu = map(self.mean, self.domain)
        K = self.cov(self.domain, self.domain, self.HP.values)

        test_x = np.array([x for x in self.domain if x not in tx])
        K_x_x = self.cov(tx, tx, self.HP.values)
        K_xs_x = self.cov(test_x, tx, self.HP.values)
        K_xs_xs = self.cov(test_x, test_x, self.HP.values)

        K_x_xs = K_xs_x.T

        c_and_lower = scipy.linalg.cho_factor(
            K_x_x, lower=False, overwrite_a=False)
        kxx_f = scipy.linalg.cho_solve(c_and_lower, ty, overwrite_b=False)

        mu = np.matmul(K_xs_x, kxx_f)

        if use_least_sq:
            inv = np.linalg.lstsq(K_x_x, K_x_xs)[0]
            K = K_xs_xs - np.matmul(K_xs_x, inv)
        else:
            # Otherwise, use a full inverse call (slow!)
            Kxx_inv = scipy.linalg.inv(K_x_x)
            K = K_xs_xs - np.matmul(K_xs_x, np.matmul(Kxx_inv, K_x_xs))

        self.mu = mu
        self.K = K

    def calculate_expected_improvement(self, domain):
        pass


if __name__ == "__main__":
    domain = np.arange(-5.1, 5.1, 0.01)
    opt = GPR_BO(domain)
    # Using a simple test function, generate training data
    train_x = np.arange(-5.1, 5.1, 1.0)
    train_y = np.array(map(f1, train_x))
    opt.fit_hp(train_x, train_y)
    opt.train(train_x, train_y)

    # Repeatedly (100 times) sample the area, using some acquisition
    # function, and train off the newly sampled points.  Each iteration,
    # save the mean and standard deviations.
    means = [copy.deepcopy(opt.mu)]
    stds = [np.diag(opt.K)]
    for i in range(10):
        EI = opt.calculate_expected_improvement(domain)
        sample_x = domain[np.nanargmax(EI)]
        sample_y = f1(sample_x)
        opt.train(sample_x, sample_y)

        means.append(copy.deepcopy(opt.mu))
        stds.append(np.diag(opt.K))

    print("Finished!")
