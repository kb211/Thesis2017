import numpy as np

class bayesnet:

    def __init__(self):

        """
        Initializer.

        """

        self.x_given_f = np.array([])
        self.p_f = np.array([])

        self.losses_nonfraud = []
        self.losses_fraud = []


    def fit(self, x_train, y_train):
        """
        Parameter Estimation

        :param x_train: Training inputs (n x |X|)
        :param y_train: Training Labels (n x |F|)
        """
        assert not np.isnan(x_train).any(), 'Input array x_train contains nan'
        assert not np.isnan(y_train).any(), 'Input array y_train contains nan'
        assert x_train.min() >= 0 and x_train.max() <= 1, 'Input x_train cointains unnormalized values'

        y_train = self.label_transform(y_train)

        self.x_given_f, self.p_f = self.mle(x_train, y_train, 0)

    def bernouli(self, theta, x):
        result = np.zeros(x.shape)

        for i in range(x.shape[1]):
            result[:, i] = np.where(x[:, i] == 1, np.log(theta[i]), np.log(1 - theta[i]))
        return result

    def predict_proba(self, x_test):

        """
        Predict y_hat for inputs X

        :param x_test: Input parameters (n x |X|)
        :return: Predictions p(F|X) (n x |F|)
        """
        assert not np.isnan(x_test).any(), 'Input array x contains nan'
        assert x_test.shape[1] == self.x_given_f.shape[0], 'Input array is of shape ' + str(x_test.shape[1]) + 'when shape (n, ' + str(self.x_given_c_f.shape[0]) + ') was expected.'
        assert x_test.min() >= 0 and x_test.max() <= 1, 'Input x_train cointains unnormalized values'

        theta_F, theta_T = self.x_given_f, self.p_f

        y_ = np.zeros((x_test.shape[0], theta_F.shape[1]))

        for t_value in range(theta_F.shape[1]):
            p_vis_0 = self.bernouli(theta_F[:, t_value], x_test)
            p_x_given_y = np.sum(p_vis_0, 1)
            y_[:, t_value] = np.log(theta_T[t_value]) + p_x_given_y

        predictions = np.exp(y_) / np.sum(np.exp(y_), axis=1, keepdims=True)

        return predictions

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

    def predict_log_proba(self, x):
        return np.log(self.predict_proba(x))










