import numpy as np
import em as em
import matplotlib.pyplot as plt

class bayesnet:

    def __init__(self):

        self.x_given_f = np.array([])
        self.p_f = np.array([])

        self.x_given_c_nonfraud = np.array([])
        self.p_c_nonfraud = np.array([])

        self.x_given_c_fraud = np.array([])
        self.p_c_fraud = np.array([])

        self.losses_nonfraud = []
        self.losses_fraud = []

        self.marginal_x_fraud = np.array([])

    def print_theta(self, theta_T, theta_F):
        for k in range(theta_T.shape[0]):
            print "t" + str(k) + ": " + str(theta_T[k])

        for i in range(theta_F.shape[0]):
            for j in range(theta_F.shape[1]):
                print "E" + str(i) + " T=" + str(j) + "\ p0: " + str(1 - theta_F[i, j]) + " p1: " + str(theta_F[i, j])


    def fit(self, x_train, y_train, n_clusters=2, epochs=20, init='random', verbose = 0):
        expmax = em.expectation_maximization(verbose, 0)

        x_train_fraud = x_train[np.argmax(y_train, axis=1) == 1, :]
        y_train_fraud = y_train[np.argmax(y_train, axis=1) == 1, :]

        x_train_nonfraud = x_train[np.argmax(y_train, axis=1) == 0, :]
        y_train_nonfraud = y_train[np.argmax(y_train, axis=1) == 0, :]


        if init == 'uniform':
            customer_f = np.ones((y_train_fraud.shape[0], n_clusters), dtype=np.float64) * (1/float(n_clusters))
            customer_n = np.ones((y_train_nonfraud.shape[0], n_clusters), dtype=np.float64) * (1 / float(n_clusters))
        elif init == 'random':
            customer_f = np.random.rand(y_train_fraud.shape[0],n_clusters)
            customer_f = customer_f/customer_f.sum(axis=1)[:,np.newaxis]

            customer_n = np.random.rand(y_train_nonfraud.shape[0],n_clusters)
            customer_n = customer_n/customer_n.sum(axis=1)[:,np.newaxis]

        self.p_c_nonfraud, self.x_given_c_nonfraud, losses_nonfraud, initial_values_nonfraud = expmax.em_algorithm(customer_n, x_train_nonfraud, epochs)
        self.p_c_fraud, self.x_given_c_fraud, losses_fraud, initial_values_fraud = expmax.em_algorithm(customer_f, x_train_fraud, epochs)
        self.x_given_f, self.p_f = self.mle(x_train, y_train, 0)

        self.marginal_x_fraud = np.mean(x_train_fraud, axis=0)

        if verbose == 1:
            self.visualize_likelihood(losses_nonfraud, color='b')
            self.visualize_likelihood(losses_fraud, color='r')

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

        # count positive examples and negative examples
        pos = features[targets[:, 1] == 1, :]
        pos_len = pos.shape[0]
        neg_len = features.shape[0] - pos_len
        pos = sum(pos)
        neg = sum(features[targets[:, 0] == 1, :])

        return np.asmatrix([((neg + alpha)/ (neg_len + alpha)), ((pos + alpha) / (pos_len + alpha))]).transpose(), [float(neg_len + alpha) / (pos_len + neg_len + alpha * features.shape[1]), float(pos_len + alpha) / (pos_len + neg_len + alpha * features.shape[1])]


    def predict_with_c(self, x_test, y_test):

        p_c, x_given_c, x_given_f, p_f = self.p_c_nonfraud, self.x_given_c_nonfraud, self.marginal_x_fraud, np.array(self.p_f)

        p_cf = np.array(p_c)[:, np.newaxis]*np.array(p_f)[:, np.newaxis].T

        p_xfraud = lambda x: np.prod(x*x_given_f + np.abs(x-1)* np.abs(x_given_f -1))

        p_xnonfraud = lambda x, c: np.prod(x*x_given_c[:, c] + np.abs(x-1)* np.abs(x_given_c[:, c]-1))

        p_x_given_c_f = lambda x, c, f: (p_xfraud(x) if f == 1 else p_xnonfraud(x, c))

        f_given_x = np.zeros((y_test.shape[0], 2))
        for i, tx in enumerate(x_test):
            joint = np.zeros((p_f.shape[0], x_given_c.shape[1]))
            for c in np.arange(x_given_c.shape[1]):
                for f in np.arange(p_f.shape[0]):
                    joint[f, c] += p_x_given_c_f(tx, c, f) * p_cf[c, f]

            f_given_x[i, :] = (np.sum(joint, axis=1) / np.sum(joint))

        return f_given_x

    def predict_c1_c2(self, x_test, y_test):
        p_c1, x_given_c1, p_c2, x_given_c2, x_given_f, p_f = self.p_c_nonfraud, self.x_given_c_nonfraud, self.p_c_fraud, self.x_given_c_fraud, self.x_given_f, np.array(self.p_f)

        p_c1_c2_f = np.array([np.array(p_c1)[:, np.newaxis]*np.array(p_c2)[:, np.newaxis].T * f for f in p_f])

        p_xfraud = lambda x, c: np.prod(x*x_given_c2[:, c] + np.abs(x-1)* np.abs(x_given_c2[:, c]-1))

        p_xnonfraud = lambda x, c: np.prod(x*x_given_c1[:, c] + np.abs(x-1)* np.abs(x_given_c1[:, c]-1))

        p_x_given_c1_c2 = lambda x, c1, c2, f: (p_xfraud(x, c1) if f == 1 else p_xnonfraud(x, c2))

        f_given_x = np.zeros((y_test.shape[0], 2))
        for i, tx in enumerate(x_test):
            joint = np.zeros((x_given_f.shape[1], x_given_c1.shape[1], x_given_c2.shape[1]))
            for c1 in np.arange(x_given_c1.shape[1]):
                for c2 in np.arange(x_given_c2.shape[1]):
                    for f in np.arange(x_given_f.shape[1]):
                        joint[f, c1, c2] += p_x_given_c1_c2(tx, c1, c2, f) * p_c1_c2_f[f, c1, c2]

            f_given_x[i, :] = (np.sum(np.sum(joint, axis=1), axis=1) / np.sum(joint))

        return f_given_x

    def sum_probas(self,x_given_f, p_f, p_c, x_given_c):
        Nx = 2 #number of values each dimension of x can take (binary in this case)

        #x_given_f, p_f = np.load('pickles/p_x_given_f(fraud).npy'), np.load('pickles/p_f(fraud).npy')
        #p_c, x_given_c = np.load('pickles/p_c(nonfraud).npy'), np.load('pickles/x_given_c(nonfraud).npy')

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

    def visualize_likelihood(self, log_likelihood, color):


        plt.plot(log_likelihood, c=color)
        plt.ylabel(r'$\ell ^ {(k)}$')
        plt.xlabel(r'Iteration $k$')
        plt.show()






