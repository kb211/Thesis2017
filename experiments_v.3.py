import numpy as np

def simulate(p_ids, p_f, c_given_id, p_c_f, x_given_c_f, n):

    F = np.random.choice(p_f.shape[0], n, p=p_f)

    IDs = np.random.choice(p_ids.shape[0], n, p=p_ids)

    C = np.array([np.random.choice(c_given_id.shape[0], p=c_given_id[:, ID]) for ID in IDs])

    X = np.array((np.array([x_given_c_f[:, c, f] for f, c in zip(F, C)]) > np.random.random((n, x_given_c_f.shape[0]))) * 1.0)

    return F, IDs, C, X



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

F, IDs, C, X = simulate(p_ids, p_f, c_given_id, p_c_f, x_given_c_f, 10000)

