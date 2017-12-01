import numpy as np

class expectation_maximization:

    def __init__(self, verbose, thresh):
        self.verbose = verbose
        self.thresh = thresh

    def expectation(self, transactions, thetas):

        theta_T, theta_F, c_given_id = thetas[0], thetas[1], thetas[2]
        y = np.asarray(np.zeros([transactions.shape[0], theta_F.shape[1]]))

        log_likelihood, ll = 0, []
        for t_value in range(theta_F.shape[1]):
            p_vis_0 = self.bernouli(theta_F[:, t_value], transactions)
            p_x_given_y = np.sum(p_vis_0, 1)
            ll.append(p_x_given_y)
            y[:, t_value] = np.log(theta_T[t_value]) + p_x_given_y

        y = np.exp(y)/np.sum(np.exp(y), axis=1, keepdims=True)
        for i in range(len(ll)):
            log_likelihood += np.dot(y[:, i], ll[i])

        return y, log_likelihood

    def maximization(self, targets, transactions, ids, alpha=0.0):

        theta_IDs = np.mean(ids, axis=0)

        theta_C = np.mean(targets, axis=0)
        C_given_ID = np.dot(targets.T, ids)

        C_given_ID /= np.sum(C_given_ID, axis=0)

        print C_given_ID
        X_given_C = np.dot(targets.T,transactions) + alpha
        X_given_C /= (np.sum(X_given_C, axis=1)[:,None] + alpha*2)


        return theta_C, X_given_C.T, C_given_ID

    def em_algorithm(self, t, f, ids, iterations):
        values = []
        #I = (f != -1) * 1

        theta_T, theta_F, c_given_id = self.maximization(t, f, ids, alpha=0.0001)

        init_theta_T, init_theta_F = theta_T, theta_F

        thresh = 0.0001
        llikelihood_old = -np.infty

        for i in range(iterations):

            t, llikelihood_new = self.expectation(f, [theta_T, theta_F, c_given_id])
            values.append(llikelihood_new)

            theta_T, theta_F, c_given_id = self.maximization(t, f, ids, alpha=0.000001)
            if self.verbose != 0:
                print "Run %d produced theta of:" % i
                print theta_T

            if np.abs(llikelihood_new - llikelihood_old) < self.thresh and self.thresh !=0:
                return theta_T, theta_F, values, [init_theta_T, init_theta_F]
            llikelihood_old = llikelihood_new

        return theta_T, theta_F, theta_T, theta_F,

    def bernouli(self, theta, x):
        result = np.zeros(x.shape)

        for i in range(x.shape[1]):
            result[:, i] = np.where(x[:, i]==1, np.log(theta[i]), np.log(1-theta[i]))
        return result