import argparse
import pandas as pd
from evaluation import evaluation
import math
import numpy as np
import v3
from sklearn import preprocessing


VERBOSE = 0
THRESH = 0

def preprocess():
    dataset = pd.read_csv('../Data-for-Dimebox/tier1_with_ids.csv')

    t = np.ones((dataset['target'].shape[0], 2))

    t[:, 1] = (dataset['target']).as_matrix()
    t[:, 0] = 1 - t[:, 1]

    ids = dataset['Card'].as_matrix()

    Amounts = (dataset['Amount']).as_matrix()

    dataset = dataset.drop(['target', 'Card'], axis=1)
    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    f = dataset.as_matrix()

    return f, t, ids, Amounts

def main():


    X, Y, ids, Amounts = preprocess()
    ratio = int(math.ceil(.8 * len(X)))
    x_train, y_train, x_test, y_test = X[:ratio], Y[:ratio], X[ratio:], Y[ratio:]
    ids_train, ids_test = ids[:ratio], ids[ratio:]


    le = preprocessing.LabelEncoder()
    le.fit(np.append(ids_train, "new_value"))


    ids_test = np.where(np.in1d(ids_test, ids_train), ids_test, "new_value")

    ids_train = le.transform(ids_train)
    ids_test = le.transform(ids_test)
    new_value_id = le.transform(["new_value"])

    model = v3.bayesnet(new_value_id)
    eval = evaluation(Amounts[ratio:])

    #print np.unique(ids_train).shape
    #print np.unique(ids_test).shape

    for rounds in range(1):

        print 'round: ' + str(rounds)

        #print "n_samples: ", n_samples
        model.fit(x_train, y_train, ids_train, n_clusters=FLAGS.n_clusters, epochs=40, verbose=1)

        #print "MLE + EM: "
        predictions = model.predict(x_test, ids_test)
        eval.evaluate(predictions, y_test)



    #eval.get_results().to_csv("results/" + "v3_" + str(FLAGS.n_clusters), index=False)



if __name__ == '__main__':

    #command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=int, default=VERBOSE, help='0 for silent, 1 for system status, 2 for graphical mode')
    parser.add_argument('--thresh', type=int, default=THRESH, help='sets threshold for new log likelihood to make algorithm stop. 0 means no threshold')
    parser.add_argument('--n_clusters', type=int, default=3, help='number of clusters')

    FLAGS, unparsed = parser.parse_known_args()

    main()