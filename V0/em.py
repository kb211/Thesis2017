import numpy as np

class expectation_maximization:

    def __init__(self, verbose, thresh):
        self.verbose = verbose
        self.thresh = thresh

    def expectation(self, transactions, I, thetas):

        theta_T, theta_F = thetas[0], thetas[1]
        y = np.asarray(np.zeros([transactions.shape[0], theta_F.shape[1]]))

        log_likelihood, ll = 0, []
        for t_value in range(theta_F.shape[1]):
            p_vis_0 = self.bernouli(theta_F[:, t_value], transactions)
            p_x_given_y = np.sum(p_vis_0, 1)
            ll.append(p_x_given_y)
            y[:, t_value] = np.log(theta_T[t_value]) + p_x_given_y

        log_likelihood = np.sum(np.exp(y))

        y = np.exp(y)/np.sum(np.exp(y), axis=1, keepdims=True)

        return y, log_likelihood

    def maximization(self, targets, transactions, alpha=0.0):

        theta_T = np.mean(targets, axis=0)

        theta_F = np.zeros([transactions.shape[1], targets.shape[1]])

        theta_F = (targets[:, :, None] * transactions[:, None, :]).sum(axis=0) / targets.sum(axis=0)[:, None]

        return theta_T, theta_F.T

    def em_algorithm(self, t, f, iterations):
        values = []
        I = (f != -1) * 1

        theta_T, theta_F = self.maximization(t, f, alpha=0.000001)

        init_theta_T, init_theta_F = theta_T, theta_F

        llikelihood_old = -np.infty

        for i in range(iterations):

            t, llikelihood_new = self.expectation(f, I, [theta_T, theta_F])
            values.append(llikelihood_new)

            theta_T, theta_F = self.maximization(t, f, alpha=0.0001)
            if self.verbose == 2:
                print "Run %d produced theta of:" % i
                print theta_T

            llikelihood_old = llikelihood_new

        return theta_T, theta_F, values, [init_theta_T, init_theta_F]

    def bernouli(self, theta, x):
        result = np.zeros(x.shape)

        for i in range(x.shape[1]):
            result[:, i] = np.where(x[:, i]==1, np.log(theta[i]), np.log(1-theta[i]))
        return result




