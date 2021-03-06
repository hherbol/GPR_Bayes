'''
This file contains a simple GPR Bayesian Optimizer
to predict a simple N-dim function.
'''
from gprbo.kernels.matern import maternKernel52 as mk52
from gprbo.kernels.periodic import periodicKernel as pk
from gprbo.hyperparameters import Theta
from gprbo.testFunctions.simple import f2

import os
import copy
import numpy as np
import scipy.stats
import scipy.linalg
from scipy import optimize as op

import matplotlib
# matplotlib.use("Agg")
from matplotlib import pyplot as plt


class GPR_BO:
    '''
    '''
    def __init__(self, domain, kern="mk52"):
        self.domain = np.array(domain)
        if len(self.domain.shape) == 1:
            self.domain = self.domain
        self.mean = lambda x: 0.0

        if kern == "mk52":
            self.cov = mk52
            self.HP = Theta("mk52", self.domain.shape)
        elif kern == "pk":
            self.cov = pk
            self.HP = Theta("pk", self.domain.shape)
        else:
            raise Exception("Invalid kern")
        if len(self.domain.shape) == 1:
            self.dim = 1
        else:
            self.dim = self.domain.shape[1]
        self.best = None
        self.sampled_indices = []

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

        mu = list(map(self.mean, X))
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

        self.best = max(ty)

        mu = np.array(list(map(self.mean, self.domain)))
        K = self.cov(self.domain, self.domain, self.HP.values)

        # test_x = np.array([x for x in self.domain if x not in tx])
        test_x = self.domain.copy()
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

    def draw(self, m=1):
        n = len(self.domain)
        A = np.linalg.cholesky(self.K + np.eye(n) * 1E-7)
        sample = np.random.normal(size=(n, m))
        return self.mu.reshape((-1, 1)) + A.dot(sample)

    def sample_most_uncertain(self):
        '''
        This function determines the next sample to run based on
        the most uncertain.
        '''
        return np.nanargmax(np.diag(self.K))

    def sample_expected_improvement(self):
        '''
        This function determines the next sample to run based on the
        Expected Improvement method.
        '''
        EI_list = np.array([
            (self.mu[i] - self.best) * scipy.stats.norm.cdf((self.mu[i] - self.best) / np.sqrt(self.K[i, i])) +
            np.sqrt(self.K[i, i]) * scipy.stats.norm.pdf((self.mu[i] - self.best) / np.sqrt(self.K[i, i]))
            if (self.K[i, i] > 0 and i not in self.sampled_indices) else 0
            # if (K[i, i] > 0 and i not in samples) else 0
            for i in range(self.dim)]).reshape((-1, self.dim))[0]
        next_sample = np.nanargmax(EI_list)

        if np.nanmax([EI_list[next_sample], 0]) <= 0:
            return np.random.choice([i for i in range(len(self.mu))])
            # return np.random.choice([i for i in range(len(self.mu)) if i not in samples])

        self.sampled_indices.append(next_sample)

        return next_sample

    def update(self, sx, sy, err=1E-6):
        '''
        Do a sequential bayesian posterior update, using the Kalman Filter method.
        Our Kalman Gain is defined as:

            KG = K[x, :] / (err + K[x, x])

        And the update is then:

            mu_new = mu_old + KG * (y_observered - mu_predicted_at_x)
        '''
        if sy > self.best:
            self.best = sy
        # This is stupid, figure out numpy search in array instead
        x = [abs(d - sx) < err for d in domain].index(True)
        # If the variance is 0 before we account for it, throw an error!
        assert self.K[x, x] > 0, "Error - Variance is 0!  Possibly double counted a point."

        cov_vec = self.K[x]
        mu_new = self.mu + (sy - self.mu[x]) / (self.K[x, x] + err) * cov_vec
        Sig_new = self.K - np.outer(self.K[x, :], self.K[:, x]) / (self.K[x, x] + err)

        self.mu, self.K = mu_new, Sig_new


if __name__ == "__main__":
    if not os.path.exists("imgs"):
        os.mkdir("imgs")

    f1 = lambda x: -f2(x)
    kern = "mk52"

    domain = np.arange(-5.1, 5.1, 0.1)
    opt = GPR_BO(domain, kern=kern)
    # Using a simple test function, generate training data
    opt.sampled_indices = list(np.random.choice(range(len(domain)), 2, replace=False))
    train_x = np.array(list(map(lambda x: domain[x], opt.sampled_indices)))
    train_y = np.array(list(map(f1, train_x)))

    full_sample_x = train_x.copy().tolist()
    full_sample_y = train_y.copy().tolist()

    plt.plot(domain, 0.0 * domain, color="dodgerblue", label="Mean")
    if kern == "mk52":
        len_HP_w = 2
    elif kern == "pk":
        len_HP_w = 3
    STD = np.diag(opt.cov(domain, domain, [1.0 for _ in range(len_HP_w)]))
    plt.fill_between(
        domain,
        2.0 * STD,
        - 2.0 * STD,
        alpha=0.5,
        color="dodgerblue"
    )
    plt.plot(domain, list(map(f1, domain)), color="orange", label="EXACT")
    plt.legend()
    plt.ylim(-10.0, 10.0)
    plt.savefig("imgs/prior.png")
    plt.close()

    opt.fit_hp(train_x, train_y)
    opt.train(train_x, train_y)

    plt.scatter(train_x, train_y, color="orange", label="Sampled")
    plt.plot(domain, opt.mu, color="dodgerblue", label="Mean")
    STD = np.diag(opt.K)
    plt.fill_between(
        domain,
        opt.mu + 2.0 * STD,
        opt.mu - 2.0 * STD,
        alpha=0.5,
        color="dodgerblue"
    )
    plt.plot(domain, list(map(f1, domain)), color="orange", label="EXACT")
    plt.legend(loc="upper left")
    plt.ylim(-10.0, 10.0)
    plt.savefig("imgs/%03d.png" % (0))
    plt.close()

    # Repeatedly (100 times) sample the area, using some acquisition
    # function, and train off the newly sampled points.  Each iteration,
    # save the mean and standard deviations.
    means = [copy.deepcopy(opt.mu)]
    stds = [np.diag(opt.K)]
    if not os.path.exists("imgs"):
        os.mkdir("imgs")
    for i in range(20):
        best_index = opt.sample_expected_improvement()
        # best_index = opt.sample_most_uncertain()
        sample_x = domain[best_index]
        sample_y = f1(sample_x)
        opt.update(sample_x, sample_y)

        means.append(copy.deepcopy(opt.mu))
        stds.append(np.diag(opt.K))

        full_sample_x.append(sample_x)
        full_sample_y.append(sample_y)

        # plt.scatter(full_sample_x, full_sample_y, color="orange", label="Sampled")

        plt.plot(domain, means[-1], color="dodgerblue")  #, label="%d" % i)
        plt.fill_between(
            domain,
            means[-1] + 2.0 * stds[-1],
            means[-1] - 2.0 * stds[-1],
            alpha=0.5,
            color="dodgerblue"
        )
        # plt.plot(domain, list(map(f1, domain)), color="orange", label="EXACT")

        plt.plot(domain, opt.draw(), color="red", label="Experiment")
        plt.plot(domain, opt.draw(), color="blue", label="DFT")
        plt.plot(domain, opt.draw(), color="green", label="MD")

        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     left=False,
        #     right=False,
        #     labelleft=False,
        #     labelbottom=False) # labels along the bottom edge are off
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

        # plt.axis('off')

        plt.legend(loc="upper left", fontsize=18)
        plt.ylim(-10.0, 10.0)
        plt.show()
        # break
        # plt.ylim(-10.0, 10.0)
        # plt.savefig("imgs/%03d.png" % (i + 1))
        # plt.close()

    # cmd = "convert -delay 10 -loop 0 $(ls -v imgs/*.png) output.gif"
    # os.system(cmd)

    print("Finished!")
