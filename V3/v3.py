import numpy as np
import em_v3 as em
import matplotlib.pyplot as plt
from id_encoder import IDEncoder

class bayesnet:

    def __init__(self):

        """
        Initializer.

        """

        self.p_f = np.array([])
        self.p_q = np.array([])

        self.x_given_c_f = np.array([])
        self.c_given_q = np.array([])

        self.likelihoods = []

    def fit(self, x_train, y_train, ids, k_clusters=2, epochs=20, verbose = 0):

        """
        Parameter Estimation

        :param x_train: Training inputs (n x |X|)
        :param y_train: Training Labels (n x |F|)
        :param k_clusters: Number of latent variables (|C|)
        :param epochs: Number of training iterations
        :param init: Initiaization of parameters
        :param verbose: 1 for likelihood plots, 2 for print of parameters at each iterations
        """

        assert not np.isnan(x_train).any(), 'Input array x_train contains nan'
        assert not np.isnan(y_train).any(), 'Input array y_train contains nan'
        assert x_train.min() >= 0 and x_train.max() <= 1, 'Input x_train cointains unnormalized values'

        y_train = self.label_transform(y_train)

        self.Encoder = IDEncoder()
        self.Encoder.fit(ids)

        ids = self.Encoder.transform(ids)

        expmax = em.expectation_maximization(verbose, 0)

        self.x_given_c_f, self.c_given_q, self.p_q, self.p_f, self.likelihoods = expmax.em_algorithm(y_train, x_train, ids, epochs, n=k_clusters)

        uniform = np.ones((self.c_given_q.shape[0])) / np.sum(np.ones((self.c_given_q.shape[0])))

        self.c_given_q = np.insert(self.c_given_q, self.Encoder.new_value, uniform, axis=1)

        if verbose == 1:
            self.visualize_likelihood(self.likelihoods)

    def predict_proba(self, x, ids):

        '''
        Predict y_hat for inputs X

        :param x: Input parameters (n x |X|)
        :param ids: Input ids
        :return: Predictions p(F|X) (n x |F|)
        '''

        assert not np.isnan(x).any(), 'Input array x contains nan'
        assert x.shape[1] == self.x_given_c_f.shape[0], 'Input array is of shape ' + str(x.shape[1]) + 'when shape (n, ' + str(self.x_given_c_f.shape[0]) + ') was expected.'
        assert x.min() >= 0 and x.max() <= 1, 'Input x_train cointains unnormalized values'

        ids = self.Encoder.transform(ids)

        x_given_c_f, c_given_q, p_f = self.x_given_c_f, self.c_given_q, self.p_f

        p_c = lambda id: c_given_q[:, id]

        p_c_f = lambda id: p_c(id)[:, None] * p_f[:, np.newaxis].T

        p_x_given_c_f = lambda x, c, f: np.prod(
            x * x_given_c_f[:, c, f] + np.abs(x - 1) * np.abs(x_given_c_f[:, c, f] - 1))

        f_given_x = np.zeros((x.shape[0], 2))
        for i, transaction in enumerate(zip(x, ids)):
            tx, ID = transaction
            joint = np.zeros((x_given_c_f.shape[1], x_given_c_f.shape[2]))
            for c in np.arange(x_given_c_f.shape[1]):
                for f in np.arange(x_given_c_f.shape[2]):
                    joint[c, f] += p_x_given_c_f(tx, c, f) * p_c_f(ID)[c, f]

            f_given_x[i, :] = (np.sum(joint, axis=0) / np.sum(joint))

        return f_given_x

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

    def label_transform(self, y):
        if y.ndim == 1:
            b = np.zeros((len(y), y.max()+1))
            b[np.arange(len(y)), y] = 1
            return b
        return y

    def predict(self, x, amounts=None, scale=.22):
        """
        :param x: Input parameters (n x |X|)
        :param amounts: Unnormalized transaction amounts n
        :return: Predictions
        """
        y_ = self.predict_proba(x)

        if amounts != None:
            amounts = np.array(amounts)
            assert amounts.shape[0] == x.shape[0], 'shape of transaction amounts does not match shape of x'

            thresholds = scale / amounts

            return np.where(y_ >= thresholds, 1., 0.)

        return np.argmax(y_, axis=1)

    def predict_log_proba(self, x, ids):
        return np.log(self.predict_proba(x, ids))