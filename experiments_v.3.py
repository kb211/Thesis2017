import numpy as np
import EM_v3 as em

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

    return X, F, IDs, C

def mle(X, F):
    p_f = np.mean(F, axis=0)
    x_given_f = np.dot(F.T, X)
    x_given_f /= (np.sum(x_given_f, axis=1)[:, None] )

    return p_f,  x_given_f.T

def predict(thetas, x_test, ids):
    x_given_c, x_given_f, c_given_id, p_id, p_f = thetas[0], thetas[1], thetas[2], thetas[3], thetas[4]

    p_c_id =  c_given_id*p_id

    p_xfraud = lambda x, c: np.prod(x * x_given_f[:, c] + np.abs(x - 1) * np.abs(x_given_f[:, c] - 1))

    p_xnonfraud = lambda x, c: np.prod(x * x_given_c1[:, c] + np.abs(x - 1) * np.abs(x_given_c1[:, c] - 1))


p_ids = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
p_f = np.array([.7, .3])

c_given_id = np.array([[0.1, 0.4, 0.4, 0.6, 0.3],
              [0.1, 0.4, 0.3, 0.2, 0.3],
              [0.8, 0.2, 0.3, 0.2, 0.4]])


p_c = np.sum(p_ids*c_given_id, axis=1)

p_c_f = p_c[:, np.newaxis]*p_f[:, np.newaxis].T

x_given_c_f = np.array([[[ 0.2,  0.2],
               [ 0.2,  0.2],
               [ 0.5,  0.1]],

               [[ 0.15,  0.6],
               [ 0.2,  0.1],
               [ 0.2,  0.2]],

               [[0.1, 0.1],
                [0.2, 0.6],
                [0.2, 0.2]],

               [[0.55, 0.1],
                [0.4, 0.1],
                [0.1, 0.5]]
               ])
print "marginal probability for c: "
print p_c
print "joint probability for c and f: "
print p_c_f
print "Assert that the conditional probability formulation is correct, by checking that for every f, c, the probabilities of x sum to 1."
print np.sum(x_given_c_f, axis=0)

#input = IDs, X, F
X, F, IDs, C = simulate(p_ids, p_f, c_given_id, p_c_f, x_given_c_f, 10000)

C_hidden = one_hot(np.random.choice(3, 10000, p=(np.arange(3)+7.)/np.sum(np.arange(3)+7.)))

expmax = em.expectation_maximization(0, 0)



p_c_learned, x_given_c_learned, c_given_id_learned = expmax.em_algorithm(C_hidden, X, IDs, 400)

p_f_learned, x_given_f_learned = mle(X, F, IDs, C)

print "assert that x_given_f_learned is true x_given_f"
print "true x_given_f:"
print np.sum(x_given_c_f*p_c_f, axis=1)/p_f
print "learned x_given_f:"
print x_given_f_learned

print "assert that x_given_c_learned is true x_given_c"
print "true x_given_c:"
print np.sum(x_given_c_f*p_c_f, axis=2)/p_c
print "leaned x_given_c:"
print x_given_c_learned

X_test, F_test, IDs_test, C_test = simulate(p_ids, p_f, c_given_id, p_c_f, x_given_c_f, 10000)

thetas = [x_given_c_learned, x_given_f_learned, c_given_id_learned, p_ids, p_f]
predict(thetas, X_test, IDs_test)
#y_ = predict(thetas, X_test, IDs_test)

#print x_given_c_f_learned.shape
#print p_c
#mle(X, F, IDs, C)




