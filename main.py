import matplotlib.pyplot as plt
import math
import datetime
import argparse
import pandas as pd
import numpy as np
import expectation_maximization as em
from evaluation import evaluation


VERBOSE = 0
THRESH = 0

def print_theta(theta_T, theta_F):
    for k in range(theta_T.shape[0]):
        print "t" + str(k) + ": " + str(theta_T[k])

    for i in range(theta_F.shape[0]):
        for j in range(theta_F.shape[1]):
            print "E" + str(i) + " T=" + str(j) + "\ p0: " + str(1 - theta_F[i, j]) + " p1: " + str(theta_F[i, j])

def preprocess():
    dataset = pd.read_csv('../Data-for-Dimebox/tier1_with_ids.csv')

    t = np.ones((dataset['target'].shape[0], 2))

    t[:, 1] = (dataset['target']).as_matrix()
    t[:, 0] = 1 - t[:, 1]

    ids = (dataset['Card']).as_matrix()

    Amounts = (dataset['Amount']).as_matrix()

    dataset = dataset.drop(['target', 'Card'], axis=1)
    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    f = dataset.as_matrix()

    return f, t, ids, Amounts


def fit(x_train, y_train, n_clusters=2, epochs=1, init='random'):
    expmax = em.expectation_maximization(FLAGS.verbose, 0)

    if init == 'uniform':
        customer = np.ones((y_train.shape[0], n_clusters), dtype=np.float64) * (1/float(n_clusters))
    elif init == 'random':
        customer = np.random.rand(y_train.shape[0],n_clusters)
        customer = customer/customer.sum(axis=1)[:,np.newaxis]

    theta_T_learned, theta_F_learned, values, initial_values = expmax.em_algorithm(customer, x_train, epochs)

    if FLAGS.verbose != 0:
        print 'Starting State:'
        theta_T_start, theta_F_start = expmax.maximization(customer, x_train)
        print_theta(theta_T_start, theta_F_start)
        print 'Ending State:'
        print_theta(theta_T_learned, theta_F_learned)

    if FLAGS.verbose == 2:
        clusters_train, likelihood_train = expmax.expectation(x_train, 1, [theta_T_learned, theta_F_learned])
        clusters_test, likelihood_test = expmax.expectation(x_test, 1, [theta_T_learned, theta_F_learned])

        data_train = np.concatenate((clusters_train, y_train[:, 1][:, None]), axis=1)
        data_test = np.concatenate((clusters_test, y_test[:, 1][:, None]), axis=1)

        data = np.concatenate((data_train, data_test), axis=0)
        data = np.concatenate((ids[:, None], data), axis=1)

        now = datetime.datetime.now()
        date = str(now.hour) + str(now.minute) + str(now.day) + str(now.month)
        filename = "/home/kasper/Documenten/Github2/Fraud_Detection2017/graphical_models/pickleandplots/" + date + "clusters"+ str(n_clusters) +"epochs"+ str(epochs)+ ".csv"
        np.savetxt(filename, data, fmt=['%s', '%.10f', '%.10f','%.10f','%.10f','%.10f','%.10f', '%.1f'], delimiter=",")

    plt.plot(np.arange(len(values)) ,values)
    plt.gca().set_ylabel(r'$\ell ^ {(k)}$')
    plt.gca().set_xlabel(r'Iteration $k$')
    plt.savefig('log_loss_' + str(FLAGS.n_clusters))

    return theta_T_learned, theta_F_learned

def bernouli(theta, x):
    result = np.zeros(x.shape)

    for i in range(x.shape[1]):
        result[:, i] = np.where(x[:, i] == 1, np.log(theta[i]), np.log(1 - theta[i]))
    return result

def predict(theta_F, theta_T, x_test, y_test, n_samples):

    y_ = np.zeros((x_test.shape[0], theta_F.shape[1]))

    for t_value in range(theta_F.shape[1]):
        p_vis_0 = bernouli(theta_F[:, t_value], x_test)
        p_x_given_y = np.sum(p_vis_0, 1)
        y_[:, t_value] = np.log(theta_T[t_value]) + p_x_given_y

    predictions = np.exp(y_) / np.sum(np.exp(y_), axis=1, keepdims=True)

    evaluationMLE.evaluate(predictions, y_test, n_samples)

def mle(features, targets, alpha=0.0001):

    # count positive examples and negative examples
    pos = features[targets[:, 1] == 1, :]
    pos_len = pos.shape[0]
    neg_len = features.shape[0] - pos_len
    pos = sum(pos)
    neg = sum(features[targets[:, 0] == 1, :])

    return np.asmatrix([((neg + alpha)/ (neg_len + alpha)), ((pos + alpha) / (pos_len + alpha))]).transpose(), [float(neg_len + alpha) / (pos_len + neg_len + alpha * features.shape[1]), float(pos_len + alpha) / (pos_len + neg_len + alpha * features.shape[1])]


def predict_with_c(x_test, y_test, params, n_samples):

    p_c, x_given_c, x_given_f, p_f = params[0], params[1], params[2], params[3]

    p_cf = np.array(p_c)[:, np.newaxis]*np.array(p_f)[:, np.newaxis].T

    p_xfraud = lambda x: np.prod(x*x_given_f[:, 1] + np.abs(x-1)* np.abs(x_given_f[:, 1]-1))

    p_xnonfraud = lambda x, c: np.prod(x*x_given_c[:, c] + np.abs(x-1)* np.abs(x_given_c[:, c]-1))

    p_x_given_c_f = lambda x, c, f: (p_xfraud(x) if f == 1 else p_xnonfraud(x, c))

    f_given_x = np.zeros((y_test.shape[0], 2))
    for i, tx in enumerate(x_test):
        joint = np.zeros((x_given_f.shape[1], x_given_c.shape[1]))
        for c in np.arange(x_given_c.shape[1]):
            for f in np.arange(x_given_f.shape[1]):
                joint[f, c] += p_x_given_c_f(tx, c, f) * p_cf[c, f]

        f_given_x[i, :] = (np.sum(joint, axis=1) / np.sum(joint))

    evaluationEM.evaluate(f_given_x, y_test, n_samples)

def predict_c1_c2(x_test, y_test, params, n_samples):
    p_c1, x_given_c1, p_c2, x_given_c2, x_given_f, p_f = params[0], params[1], params[2], params[3], params[4], np.array(params[5])

    p_c1_c2_f = np.array([np.array(p_c_nonfraud)[:, np.newaxis]*np.array(p_c2)[:, np.newaxis].T * f for f in p_f])

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

    evaluationEM2.evaluate(f_given_x, y_test, n_samples)

def sum_probas(x_given_f, p_f, p_c, x_given_c):
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

if __name__ == '__main__':

    #command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=int, default=VERBOSE, help='0 for silent, 1 for system status, 2 for graphical mode')
    parser.add_argument('--thresh', type=int, default=THRESH, help='sets threshold for new log likelihood to make algorithm stop. 0 means no threshold')
    parser.add_argument('--n_clusters', type=int, default=3, help='number of clusters')

    FLAGS, unparsed = parser.parse_known_args()

    X, Y, ids, Amounts = preprocess()
    ratio = int(math.ceil(.8*len(X)))
    x_train, y_train, x_test, y_test = X[:ratio], Y[:ratio], X[ratio:], Y[ratio:]

    evaluationMLE = evaluation(Amounts[ratio:])
    evaluationEM = evaluation(Amounts[ratio:])
    evaluationEM2 = evaluation(Amounts[ratio:])

    customer = np.random.rand(y_train.shape[0], FLAGS.n_clusters)
    customer = customer / customer.sum(axis=1)[:, np.newaxis]

    x_train_fraud = x_train[np.argmax(y_train, axis=1) == 1, :]
    y_train_fraud = y_train[np.argmax(y_train, axis=1) == 1, :]

    x_train_nonfraud = x_train[np.argmax(y_train, axis=1) == 0, :]
    y_train_nonfraud = y_train[np.argmax(y_train, axis=1) == 0, :]

    #p_c, x_given_c = fit(x_train_nonfraud[indexes], y_train_nonfraud[indexes], n_clusters=FLAGS.n_clusters, epochs=20, init='random')

    #np.save('p_x_given_f(fraud)', p_x_given_f)
    #np.save('p_f(fraud)', p_f)
    #np.save('pickles/p_c(nonfraud)' + str(FLAGS.n_clusters), p_c)
    #np.save('pickles/x_given_c(nonfraud)'+ str(FLAGS.n_clusters), x_given_c)

    #x_given_f, p_f = np.load('pickles/p_x_given_f.npy'), np.load('pickles/p_f.npy')


    for n_samples in [1051, 1051 * 2, 1051 * 4, 1051 * 8, 1051 * 16, 1051 * 32]:

        p_c_nonfraud = np.load('pickles/p_c(nonfraud)' + str(FLAGS.n_clusters) + "_" + str(n_samples) + '.npy')
        x_given_c_nonfraud = np.load('pickles/x_given_c(nonfraud)' + str(FLAGS.n_clusters)+ "_" + str(n_samples) + '.npy')

        p_c_fraud = np.load('pickles/p_c(fraud)' + str(FLAGS.n_clusters) + '.npy')
        x_given_c_fraud = np.load('pickles/x_given_c(fraud)' + str(FLAGS.n_clusters) + '.npy')

        indexes = np.random.choice(x_train_nonfraud.shape[0], n_samples)

        x_train_undersample = np.concatenate((x_train_nonfraud[indexes], x_train_fraud), axis=0)
        y_train_undersample = np.concatenate((y_train_nonfraud[indexes], y_train_fraud), axis=0)

        #print "n_samples: ", n_samples
        x_given_f, p_f = mle(x_train_undersample, y_train_undersample, 0)

        #print "MLE: "
        predict(x_given_f, p_f, x_test, y_test, n_samples)

        #print "MLE + EM: "
        params = [p_c_nonfraud, x_given_c_nonfraud, x_given_f, p_f]
        predict_with_c(x_test, y_test, params, n_samples)

        #print "EM + EM: "
        params2 = [p_c_nonfraud, x_given_c_nonfraud, p_c_fraud, x_given_c_fraud, x_given_f, p_f]
        predict_c1_c2(x_test, y_test, params2, n_samples)

    evaluationMLE.get_results().to_csv("results/" + "MLE_" + str(FLAGS.n_clusters), index=False)
    evaluationEM.get_results().to_csv("results/" + "EMandMLE_"+ str(FLAGS.n_clusters), index=False)
    evaluationEM2.get_results().to_csv("results/" + "EMandEM_"+ str(FLAGS.n_clusters), index=False)

