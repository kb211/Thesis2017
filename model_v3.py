import numpy as np
import EM_v3 as em
import matplotlib.pyplot as plt

class bayesnet:

    def __init__(self, new_value_id):

        self.new_value_id = new_value_id

        self.p_f = np.array([])
        self.p_q = np.array([])

        self.x_given_c_f = np.array([])
        self.c_given_q = np.array([])

        self.losses_nonfraud = []
        self.losses_fraud = []



    def normalize(self, arr, axis=None):
        arr = np.array(arr, copy=False, dtype=np.float)
        assert arr.ndim > 0, 'This makes no sense on scalars'
        if axis is None:
            assert arr.ndim == 1, 'If array dim is not 1 you must specify axis'
            axis = 0
        assert np.all(arr >= 0)
        return arr / np.sum(arr, axis=axis, keepdims=True)

    def print_theta(self, theta_T, theta_F):
        for k in range(theta_T.shape[0]):
            print "t" + str(k) + ": " + str(theta_T[k])

        for i in range(theta_F.shape[0]):
            for j in range(theta_F.shape[1]):
                print "E" + str(i) + " T=" + str(j) + "\ p0: " + str(1 - theta_F[i, j]) + " p1: " + str(theta_F[i, j])


    def fit(self, x_train, y_train, ids, n_clusters=2, epochs=20, init='random', verbose = 0):
        expmax = em.expectation_maximization(verbose, 0)

        self.x_given_c_f, self.c_given_q, self.p_q, self.p_f, likelihoods = expmax.em_algorithm(y_train, x_train, ids, epochs, n=n_clusters)

        uniform = np.ones((self.c_given_q.shape[0], 1)) / np.sum(np.ones((self.c_given_q.shape[0], 1)))

        self.c_given_q = np.append(self.c_given_q, uniform, axis=1)

        if verbose == 1:
            self.visualize_likelihood(likelihoods)

    def bernouli(self, theta, x):
        result = np.zeros(x.shape)

        for i in range(x.shape[1]):
            result[:, i] = np.where(x[:, i] == 1, np.log(theta[i]), np.log(1 - theta[i]))
        return result

    def predict(self, x, ids):
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

    def visualize_likelihood(self, log_likelihood, color='b'):


        plt.plot(log_likelihood, c=color)
        plt.ylabel(r'$\ell ^ {(k)}$')
        plt.xlabel(r'Iteration $k$')
        plt.show()




