import numpy as np
import expectation_maximization as em

def one_hot(array):
    out = np.zeros((array.shape[0], np.max(array)+1))
    out[np.arange(array.shape[0]), array] = 1
    return out


def simulate(p_ids, p_f, c_given_id, p_c_f, x_given_c_f, n):

    F = np.random.choice(p_f.shape[0], n, p=p_f)

    IDs = np.random.choice(p_ids.shape[0], n, p=p_ids)

    C = np.array([np.random.choice(c_given_id.shape[0], p=c_given_id[:, ID]) for ID in IDs])

    X = np.array((np.array([x_given_c_f[:, c, f] for f, c in zip(F, C)]) > np.random.random((n, x_given_c_f.shape[0]))) * 1.0)

    IDs = one_hot(IDs)
    F = one_hot(F)
    C = one_hot(C)

    input = np.concatenate((np.concatenate((IDs,X), axis=1), F), axis=1)
    return input, C



p_ids = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
p_f = np.array([.7, .3])

c_given_id = np.array([[0.1, 0.4, 0.4, 0.6, 0.3],
              [0.1, 0.4, 0.3, 0.2, 0.3],
              [0.8, 0.2, 0.3, 0.2, 0.4]])

p_c = np.sum(p_ids*c_given_id, axis=1)

p_c_f = p_c[:, np.newaxis]*p_f[:, np.newaxis].T

x_given_c_f = np.array([[[ 0.2,  0.2],
               [ 0.2,  0.2],
               [ 0.1,  0.1]],

               [[ 0.15,  0.15],
               [ 0.2,  0.1],
               [ 0.2,  0.2]],

               [[0.1, 0.1],
                [0.2, 0.15],
                [0.2, 0.25]],

               [[0.1, 0.1],
                [0.1, 0.1],
                [0.1, 0.5]]
               ])
print "marginal probability for c: "
print p_c
print "joint probability for c and f: "
print p_c_f
print "Assert that the conditional probability formulation is correct, by checking that for every f, c, the probabilities of x sum to 1."
print np.sum(np.sum(x_given_c_f, axis=1), axis=1)

#input = IDs, X, F
input, C = simulate(p_ids, p_f, c_given_id, p_c_f, x_given_c_f, 10000)

C_hidden = one_hot(np.random.choice(3, 10000, p=(np.arange(3)+1.)/np.sum(np.arange(3)+1.)))

expmax = em.expectation_maximization(1, 0)

p_c_nonfraud, x_given_c_nonfraud, losses_nonfraud, initial_values_nonfraud = expmax.em_algorithm(C_hidden, input, 200)




