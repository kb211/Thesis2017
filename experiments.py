import argparse
import pandas as pd
from evaluation import evaluation
import math
import numpy as np
import bayesian_network as bn
import matplotlib.pyplot as plt
import datetime

VERBOSE = 0
THRESH = 0

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

def main():
    model = bn.bayesnet()

    X, Y, ids, Amounts = preprocess()
    ratio = int(math.ceil(.8 * len(X)))
    x_train, y_train, x_test, y_test = X[:ratio], Y[:ratio], X[ratio:], Y[ratio:]

    evaluationMLE = evaluation(Amounts[ratio:])
    evaluationEM = evaluation(Amounts[ratio:])
    evaluationEM2 = evaluation(Amounts[ratio:])

    print "MLE: "
    model.x_given_f, model.p_f = model.mle(x_train, y_train)
    predictions_MLE = model.predict(x_test, y_test)
    evaluationMLE.evaluate(predictions_MLE, y_test)
    '''
    for rounds in range(100):
        if np.remainder(rounds, 10) == 0:
            print 'round: ' + str(rounds)

        # print "n_samples: ", n_samples
        model.fit(x_train, y_train, n_clusters=FLAGS.n_clusters, epochs=20)

        # print "MLE + EM: "
        predictions_c1 = model.predict_with_c(x_test, y_test)
        evaluationEM.evaluate(predictions_c1, y_test)

        # print "EM + EM: "
        predictions_c1_c2 = model.predict_c1_c2(x_test, y_test)
        evaluationEM2.evaluate(predictions_c1_c2, y_test)
    '''

    evaluationMLE.get_results().to_csv("results/" + "MLE_" + str(FLAGS.n_clusters), index=False)
    #evaluationEM.get_average_scores().to_csv("results/" + "EMandMLE_" + str(FLAGS.n_clusters), index=False)
    #evaluationEM2.get_average_scores().to_csv("results/" + "EMandEM_" + str(FLAGS.n_clusters), index=False)



if __name__ == '__main__':

    #command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=int, default=VERBOSE, help='0 for silent, 1 for system status, 2 for graphical mode')
    parser.add_argument('--thresh', type=int, default=THRESH, help='sets threshold for new log likelihood to make algorithm stop. 0 means no threshold')
    parser.add_argument('--n_clusters', type=int, default=3, help='number of clusters')

    FLAGS, unparsed = parser.parse_known_args()

    main()

'''

    # p_c, x_given_c = fit(x_train_nonfraud[indexes], y_train_nonfraud[indexes], n_clusters=FLAGS.n_clusters, epochs=20, init='random')

    # np.save('p_x_given_f(fraud)', p_x_given_f)
    # np.save('p_f(fraud)', p_f)
    # np.save('pickles/p_c(nonfraud)' + str(FLAGS.n_clusters), p_c)
    # np.save('pickles/x_given_c(nonfraud)'+ str(FLAGS.n_clusters), x_given_c)

    # x_given_f, p_f = np.load('pickles/p_x_given_f.npy'), np.load('pickles/p_f.npy')


    for n_samples in [1051 * 16, 1051 * 32, math.ceil(x_train_nonfraud.shape[0]*0.5), x_train_nonfraud.shape[0]]:

        p_c_nonfraud = np.load('pickles/p_c(nonfraud)' + str(FLAGS.n_clusters) + "_" + str(n_samples) + '.npy')
        x_given_c_nonfraud = np.load(
            'pickles/x_given_c(nonfraud)' + str(FLAGS.n_clusters) + "_" + str(n_samples) + '.npy')

        p_c_fraud = np.load('pickles/p_c(fraud)' + str(FLAGS.n_clusters) + '.npy')
        x_given_c_fraud = np.load('pickles/x_given_c(fraud)' + str(FLAGS.n_clusters) + '.npy')

        indexes = np.random.choice(x_train_nonfraud.shape[0], n_samples)
        x_train_undersample = np.concatenate((x_train_nonfraud[indexes], x_train_fraud), axis=0)
        y_train_undersample = np.concatenate((y_train_nonfraud[indexes], y_train_fraud), axis=0)
'''