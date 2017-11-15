import numpy as np

class data_generator:

    def __init__(self, t, f, fc):
        self.t = t
        self.f = f
        self.fc = fc

    def simulate(self, n, distr='binom'):

        T = np.zeros([2, n])
        T[1] = (self.t[1] > np.random.random(n))
        T[0] = 1 - T[1]


        if distr == 'binom':
            F = (np.asmatrix(T[0]).transpose() * self.f[:, 0] > np.random.random([n, self.f.shape[0]])) \
                + (np.asmatrix(T[1]).transpose() * self.f[:, 1] > np.random.random([n, self.f.shape[0]]))

            F = np.asarray(F * 1.)

            FC = (np.asmatrix(T[0]).transpose() * self.fc[:, 0] > np.random.random([n, self.fc.shape[0]])) \
                + (np.asmatrix(T[1]).transpose() * self.fc[:, 1] > np.random.random([n, self.fc.shape[0]]))

            FC = np.asarray(FC * 1.)
        else:
            # create some continuous variables with mu = theta_fc and std = 0.1
            F = np.asarray([np.random.normal(mu, 1, n) for mu in self.f[:, 0]]).T * np.array(T[0])[:, np.newaxis] \
                 + np.asarray([np.random.normal(mu, 1, n) for mu in self.f[:, 1]]).T * np.array(T[1])[:, np.newaxis]



            FC = np.asarray([np.random.normal(mu, 1, n) for mu in self.fc[:, 0]]).T * np.array(T[0])[:, np.newaxis] \
                 + np.asarray([np.random.normal(mu, 1, n) for mu in self.fc[:, 1]]).T * np.array(T[1])[:, np.newaxis]

        return T.T, np.concatenate((F, FC), axis=1)