import numpy as np
import em as em
import matplotlib.pyplot as plt

class bayesnet:

    def __init__(self):

        self.x_given_f = np.array([])
        self.p_f = np.array([])

        self.losses_nonfraud = []
        self.losses_fraud = []


    def fit(self, x_train, y_train, n_clusters=2, epochs=20, init='random', verbose = 0):

        self.x_given_f, self.p_f = self.mle(x_train, y_train, 0)

        #if verbose == 1:
        #    self.visualize_likelihood(losses_nonfraud, color='b')
        #    self.visualize_likelihood(losses_fraud, color='r')

    def bernouli(self, theta, x):
        result = np.zeros(x.shape)

        for i in range(x.shape[1]):
            result[:, i] = np.where(x[:, i] == 1, np.log(theta[i]), np.log(1 - theta[i]))
        return result

    def predict(self, x_test, y_test):
        theta_F, theta_T = self.x_given_f, self.p_f

        y_ = np.zeros((x_test.shape[0], theta_F.shape[1]))

        for t_value in range(theta_F.shape[1]):
            p_vis_0 = self.bernouli(theta_F[:, t_value], x_test)
            p_x_given_y = np.sum(p_vis_0, 1)
            y_[:, t_value] = np.log(theta_T[t_value]) + p_x_given_y

        predictions = np.exp(y_) / np.sum(np.exp(y_), axis=1, keepdims=True)

        return predictions

    def mle(self, features, targets, alpha=0.0001):
        '''
        :param features: Training inputs
        :param targets: Training labels
        :param alpha:
        :return:
        '''
        # count positive examples and negative examples
        pos = features[targets[:, 1] == 1, :]
        pos_len = pos.shape[0]
        neg_len = features.shape[0] - pos_len
        pos = sum(pos)
        neg = sum(features[targets[:, 0] == 1, :])

        return np.asmatrix([((neg + alpha)/ (neg_len + alpha)), ((pos + alpha) / (pos_len + alpha))]).transpose(), [float(neg_len + alpha) / (pos_len + neg_len + alpha * features.shape[1]), float(pos_len + alpha) / (pos_len + neg_len + alpha * features.shape[1])]


    def visualize_likelihood(self, log_likelihood, color):


        plt.plot(log_likelihood, c=color)
        plt.ylabel(r'$\ell ^ {(k)}$')
        plt.xlabel(r'Iteration $k$')
        plt.show()






