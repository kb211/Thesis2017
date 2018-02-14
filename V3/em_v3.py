import numpy as np

class expectation_maximization:

    def __init__(self, verbose, thresh):
        self.verbose = verbose
        self.thresh = thresh

    def one_hot(self, array):
        out = np.zeros((array.shape[0], np.max(array) + 1))
        out[np.arange(array.shape[0]), array] = 1
        return out

    def normalize(self, arr, axis=None):
        arr = np.array(arr, copy=False, dtype=np.float)
        assert arr.ndim > 0, 'This makes no sense on scalars'
        if axis is None:
            assert arr.ndim == 1, 'If array dim is not 1 you must specify axis'
            axis = 0
        assert np.all(arr >= 0)
        return arr / np.sum(arr, axis=axis, keepdims=True)

    def expectation(self, f, x, ids, thetas):

        x_given_c_f, c_given_q = thetas[0], thetas[1]

        p_c = c_given_q[:, ids].T

        x_given_c = x_given_c_f[:, :, np.argmax(f, axis=1)].T
        intermediate = (np.log(x_given_c) * x[:, None, :]) + (np.log(1 - x_given_c) * (1 - x[:, None, :]))

        p_x = np.log(p_c) + np.sum(intermediate, axis=2)

        assert not np.isnan(p_x).any()
        likelihood = np.sum(np.exp(p_x))
        y = np.exp(p_x) / np.sum(np.exp(p_x), axis=1, keepdims=True)

        assert not np.isnan(y).any()
        return y, likelihood

    def maximization(self, f, x, ids, c, alpha=0.0):

        x_given_c_f = ((f[:, :, None]* c[:, None, :])[:,:,:,None]*x[:, None, None, :] + alpha).sum(axis=0) / (f[:, :, None]* c[:, None, :] + (alpha*2)).sum(axis=0)[:, :, None]

        c_and_id = np.concatenate([c, ids[:, None]], axis=1)
        sorted = c_and_id[c_and_id[:, -1].argsort()]
        splitted = np.split(sorted, np.where(np.diff(sorted[:, c.shape[1]]))[0] + 1)
        c_given_q2 = np.zeros((c.shape[1], np.max(ids)+1))

        for split in splitted:
            idx = int(split[0, -1])
            c_given_q2[:, idx] = (np.sum(split, axis=0)[:-1] + alpha) / (split.shape[1] + (alpha*2))

        assert not np.isnan(x_given_c_f.T).any()
        assert not np.isnan(c_given_q2).any()

        return x_given_c_f.T, c_given_q2

    def em_algorithm(self, f, x, ids, iterations, n=3):
        likelihood_values = []

        x_given_c_f = self.normalize(np.random.rand(x.shape[1], n, f.shape[1]), axis=0)
        c_given_q = self.normalize(np.random.rand(n, np.unique(ids).shape[0]), axis=0)
        p_q = self.normalize(np.bincount(ids))
        p_f = np.mean(f, axis=0)

        likelihood_old = -np.infty
        for i in range(iterations):

            thetas = [x_given_c_f, c_given_q]

            c, likelihood_new = self.expectation(f, x, ids, thetas)
            likelihood_values.append(likelihood_new)
            x_given_c_f, c_given_q = self.maximization(f, x, ids, c, alpha=0.000001)

        likelihood_old = likelihood_new

        return x_given_c_f, c_given_q, p_q, p_f, likelihood_values

    def bernouli(self, theta, x):
        result = np.zeros(x.shape)

        for i in range(x.shape[1]):
            result[:, i] = np.where(x[:, i]==1, np.log(theta[i]), np.log(1-theta[i]))
        return result