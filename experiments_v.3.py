import numpy as np
import EM_v3 as em

def one_hot(array):
    out = np.zeros((array.shape[0], np.max(array)+1))
    out[np.arange(array.shape[0]), array] = 1
    return out

def evaluate(y_, y):

    precision = lambda tp, fp: float(tp) / (tp + fp) if (tp + fp) > 0 else 0
    recall = lambda tp, fn: float(tp) / (tp + fn) if (tp + fn) > 0 else 0
    specificity = lambda tn, fp: float(tn) / (tn + fp) if (tn + fp) > 0 else 0
    tp, tn, fp, fn = 0, 0, 0, 0

    for prediction, label in zip(y_, y):

        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and  prediction == 0:
            tn += 1
        elif label == 1 and prediction == 0:
            fn += 1
        else:
            fp += 1

    r = recall(tp, fn)
    p = precision(tp, fp)
    s = specificity(tn, fp)
    print tp, tn, fp, fn
    return r, p, s

def simulate(p_ids, p_f, c_given_id, p_c_f, x_given_c_f, n):

    F = np.random.choice(p_f.shape[0], n, p=p_f)

    IDs = np.random.choice(p_ids.shape[0], n, p=p_ids)

    C = np.array([np.random.choice(c_given_id.shape[0], p=c_given_id[:, ID]) for ID in IDs])

    X = np.array((np.array([x_given_c_f[:, c, f] for f, c in zip(F, C)]) > np.random.random((n, x_given_c_f.shape[0]))) * 1.0)

    #IDs = one_hot(IDs)
    F = one_hot(F)
    C = one_hot(C)

    return X, F, IDs, C

def mle(X, F):
    p_f = np.mean(F, axis=0)
    x_given_f = np.dot(F.T, X)
    x_given_f /= (np.sum(x_given_f, axis=1)[:, None] )

    return p_f,  x_given_f.T

def predict(thetas, x, ids):
    x_given_c_f, c_given_q, p_f = thetas[0], thetas[1], thetas[2]

    p_c = lambda id: c_given_q[:, id]

    p_c_f = lambda id: p_c(id)[:, None]* p_f[:, np.newaxis].T

    p_x_given_c_f = lambda x, c, f: np.prod(x * x_given_c_f[:, c, f] + np.abs(x - 1) * np.abs(x_given_c_f[:, c, f] - 1))

    f_given_x = np.zeros((x.shape[0], 2))
    for i, transaction in enumerate(zip(x, ids)):
        tx, ID = transaction
        joint = np.zeros((x_given_c_f.shape[1], x_given_c_f.shape[2]))
        for c in np.arange(x_given_c_f.shape[1]):
            for f in np.arange(x_given_c_f.shape[2]):
                joint[c, f] += p_x_given_c_f(tx, c, f) * p_c_f(ID)[c, f]

        f_given_x[i, :] = (np.sum(joint, axis=0) / np.sum(joint))

    return f_given_x

def predict2(thetas, x, ids):
    x_given_c_f, c_given_q, p_f = thetas[0], thetas[1], thetas[2]


    p_c = c_given_q[:, ids].T

    x_given_f = x_given_c_f[None, :, :, :] * p_c[:, None, :, None]
    joint =  x_given_f *p_f[None, None, None, :]


    joint = np.prod(((joint) * x[:, :, None, None]) + ((1 - joint) * (1 - x[:, :, None, None])), axis=1)


    y = np.sum(joint, axis=1)
    y = y / np.sum(y, axis=1, keepdims=True)

    return y



p_ids = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
p_f = np.array([.6, .4])

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
print
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

x_given_f_c_learned, c_given_id_learned, p_q_learned, p_f_learned = expmax.em_algorithm(F, X, IDs, 40, n=3)

#print x_given_f_c_learned
#print c_given_id_learned


X_test, F_test, IDs_test, C_test = simulate(p_ids, p_f, c_given_id, p_c_f, x_given_c_f, 10000)


thetas = [x_given_f_c_learned, c_given_id_learned, p_f_learned]
y_ = predict(thetas, X, IDs)

print np.mean(F, axis=0)
print np.mean(y_, axis=0)
print evaluate(np.argmax(y_, axis=1), np.argmax(F, axis=1))


thetas2 = [x_given_c_f, c_given_id, p_f]
y_ = predict(thetas2, X_test, IDs_test)

print np.mean(F_test, axis=0)
print np.mean(y_, axis=0)
print evaluate(np.argmax(y_, axis=1), np.argmax(F_test, axis=1))

print F_test, y_

'''
y2 = predict2(thetas, X_test, IDs_test)
print np.mean(F_test, axis=0)
print np.mean(y2, axis=0)
print evaluate(np.argmax(y2, axis=1), np.argmax(F_test, axis=1))


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


'''
#y_ = predict(thetas, X_test, IDs_test)

#print x_given_c_f_learned.shape
#print p_c
#mle(X, F, IDs, C)




