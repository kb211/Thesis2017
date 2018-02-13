import numpy as np
import em as em
import matplotlib.pyplot as plt

class bayesnet:

    def __init__(self):

        """
        Initializer.

        """

        self.x_given_f = np.array([])
        self.p_f = np.array([])

        self.x_given_c_nonfraud = np.array([])
        self.p_c_nonfraud = np.array([])

        self.losses_nonfraud = []
        self.losses_fraud = []

        self.marginal_x_fraud = np.array([])

    def fit(self, x_train, y_train, k_clusters=2, epochs=20, init='random', verbose = 0):
        """
        Parameter Estimation

        :param x_train: Training inputs (n x |X|)
        :param y_train: Training Labels (n x |F|)
        :param k_clusters: Number of latent variables (|C|)
        :param epochs: Number of training iterations
        :param init: Initiaization of parameters
        :param verbose: 1 for likelihood plots, 2 for print of parameters at each iterations
        """

        expmax = em.expectation_maximization(verbose, 0)

        x_train_fraud = x_train[np.argmax(y_train, axis=1) == 1, :]
        y_train_fraud = y_train[np.argmax(y_train, axis=1) == 1, :]

        x_train_nonfraud = x_train[np.argmax(y_train, axis=1) == 0, :]
        y_train_nonfraud = y_train[np.argmax(y_train, axis=1) == 0, :]


        if init == 'uniform':
            customer_f = np.ones((y_train_fraud.shape[0], k_clusters), dtype=np.float64) * (1/float(k_clusters))
            customer_n = np.ones((y_train_nonfraud.shape[0], k_clusters), dtype=np.float64) * (1 / float(k_clusters))
        elif init == 'random':
            customer_f = np.random.rand(y_train_fraud.shape[0],k_clusters)
            customer_f = customer_f/customer_f.sum(axis=1)[:,np.newaxis]

            customer_n = np.random.rand(y_train_nonfraud.shape[0],k_clusters)
            customer_n = customer_n/customer_n.sum(axis=1)[:,np.newaxis]

        self.p_c_nonfraud, self.x_given_c_nonfraud, losses_nonfraud, initial_values_nonfraud = expmax.em_algorithm(customer_n, x_train_nonfraud, epochs)
        self.x_given_f, self.p_f = self.mle(x_train, y_train, 0)

        self.marginal_x_fraud = np.mean(x_train_fraud, axis=0)

        if verbose == 1:
            self.visualize_likelihood(losses_nonfraud, color='b')

    def mle(self, features, targets, alpha=0.0001):
        """
        Maximum likelihood estimation

        :param features: Training inputs (n x |X|)
        :param targets: Training labels (n x |F|)
        :param alpha: Smoothing parameter for laplace smoothing
        :return: X given F and probability of F p(X|F), p(F)
        """

        # count positive examples and negative examples
        pos = features[targets[:, 1] == 1, :]
        pos_len = pos.shape[0]
        neg_len = features.shape[0] - pos_len
        pos = sum(pos)
        neg = sum(features[targets[:, 0] == 1, :])

        return np.asmatrix([((neg + alpha)/ (neg_len + alpha)), ((pos + alpha) / (pos_len + alpha))]).transpose(), [float(neg_len + alpha) / (pos_len + neg_len + alpha * features.shape[1]), float(pos_len + alpha) / (pos_len + neg_len + alpha * features.shape[1])]


    def predict(self, x_test):
        """
        Predict y_hat for inputs X

        :param x_test: Input parameters (n x |X|)
        :return: Predictions p(F|X) (n x |F|)
        """

        p_c, x_given_c, x_given_f, p_f = self.p_c_nonfraud, self.x_given_c_nonfraud, self.marginal_x_fraud, np.array(self.p_f)

        p_cf = np.array(p_c)[:, np.newaxis]*np.array(p_f)[:, np.newaxis].T

        p_xfraud = lambda x: np.prod(x*x_given_f + np.abs(x-1)* np.abs(x_given_f -1))

        p_xnonfraud = lambda x, c: np.prod(x*x_given_c[:, c] + np.abs(x-1)* np.abs(x_given_c[:, c]-1))

        p_x_given_c_f = lambda x, c, f: (p_xfraud(x) if f == 1 else p_xnonfraud(x, c))

        f_given_x = np.zeros((x_test.shape[0], 2))
        for i, tx in enumerate(x_test):
            joint = np.zeros((p_f.shape[0], x_given_c.shape[1]))
            for c in np.arange(x_given_c.shape[1]):
                for f in np.arange(p_f.shape[0]):
                    joint[f, c] += p_x_given_c_f(tx, c, f) * p_cf[c, f]

            f_given_x[i, :] = (np.sum(joint, axis=1) / np.sum(joint))

        return f_given_x

    def sum_probas(self,x_given_f, p_f, p_c, x_given_c):

        """
        prints the sum of the joint distribution. In order to assess whether or not it sums to one.

        :param x_given_f: p(X|F)
        :param p_f: p(F)
        :param p_c: p(C)
        :param x_given_c: p(X|C)
        """

        Nx = 2 #number of values each dimension of x can take (binary in this case)

        all_possible_cs = np.arange(x_given_c.shape[1])  # (Nc, )
        all_possible_fs = np.arange(x_given_f.shape[1])

        p_cf = np.array(p_c)[:, np.newaxis]*np.array(p_f)[:, np.newaxis].T

        print np.sum(p_cf)

        p_xifraud = lambda x, i: np.prod(x*x_given_f[i, 1] + np.abs(x-1)* np.abs(x_given_f[i, 1]-1))

        p_xinonfraud = lambda x, c, i: np.prod(x*x_given_c[i, c] + np.abs(x-1)* np.abs(x_given_c[i, c]-1))

        p_xi_given_c_f = lambda x, c, f, i: (p_xifraud(x, i) if f == 1 else p_xinonfraud(x, c, i))

        p_xi_c_f = lambda x, c, f, i: p_xi_given_c_f(x, c, f, i) * p_cf[c, f]

        table_p_x_c_f_i = np.array([[[[ p_xi_c_f(x, c, f, i) for i in range(x_given_f.shape[0])] for x in range(Nx)] for c in all_possible_cs] for f in all_possible_fs])

        print table_p_x_c_f_i.sum(axis=0).sum(axis=0).sum(axis=0)

    def visualize_likelihood(self, likelihood, color='b'):

        '''
        Visualize likelihood.

        :param likelihood: List of likelihood values
        :param color: Color of plot
        '''

        plt.plot(likelihood, c=color)
        plt.ylabel(r'$\ell ^ {(k)}$')
        plt.xlabel(r'Iteration $k$')
        plt.show()






