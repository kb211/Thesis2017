import argparse
import pandas as pd
from evaluation import evaluation
import math
import numpy as np
import v0 as bn
#import matplotlib.pyplot as plt
import datetime

VERBOSE = 0
THRESH = 0

def preprocess():
    #dataset = pd.read_csv('../Data-for-Dimebox/tier1_with_ids.csv')
    dataset = pd.read_csv('../../../datasets/Loan_payments_2.csv').sample(frac=1)

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

    eval = evaluation(Amounts[ratio:])


    print "MLE: "
    model.x_given_f, model.p_f = model.mle(x_train, y_train)
    predictions_v0 = model.predict(x_test)
    eval.evaluate(predictions_v0, y_test)

    eval.get_results().to_csv("../results/" + "v0", index=False)

    print predictions_v0, y_test

if __name__ == '__main__':

    #command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=int, default=VERBOSE, help='0 for silent, 1 for system status, 2 for graphical mode')
    parser.add_argument('--thresh', type=int, default=THRESH, help='sets threshold for new log likelihood to make algorithm stop. 0 means no threshold')
    parser.add_argument('--n_clusters', type=int, default=3, help='number of clusters')

    FLAGS, unparsed = parser.parse_known_args()

    main()

