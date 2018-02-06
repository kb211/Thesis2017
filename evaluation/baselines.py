from sklearn.ensemble import RandomForestClassifier
from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.metrics import savings_score
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from costcla.models import CostSensitiveLogisticRegression


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

def evaluate(y_, y, costmatrix):

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

    savings = savings_score(y, y_, costmatrix)

    return r, p, s, savings


X, Y, ids, Amounts = preprocess()
ratio = int(math.ceil(.8 * len(X)))
x_train, y_train, x_test, y_test = X[:ratio], Y[:ratio], X[ratio:], Y[ratio:]

# cost_mat [false positives, false negatives, true positives, true negatives]
cost_mat = np.ones((X.shape[0], 4))
cost_mat[:, 0] *= 0.22
cost_mat[:, 1] = Amounts
cost_mat[:, 2] *= 0.22
cost_mat[:, 3] *= 0

cost_mat_train, cost_mat_test = cost_mat[:ratio], cost_mat[ratio:]

y_train, y_test, = np.argmax(y_train, axis=1), np.argmax(y_test, axis=1)

print y_train.shape, y_test.shape

#random forest
rfc = RandomForestClassifier(random_state=0).fit(x_train, y_train)
y_pred_test_rf = rfc.predict(x_test)

print evaluate(y_pred_test_rf, y_test, cost_mat_test)

#logistic regression
lr = LogisticRegression(random_state=0).fit(x_train, y_train)
y_pred_test_lr = lr.predict(x_test)

print evaluate(y_pred_test_lr, y_test, cost_mat_test)

#cost-sensitive decision trees
CSDT = CostSensitiveDecisionTreeClassifier().fit(x_train, y_train, cost_mat_train)
y_pred_test_csdt = CSDT.predict(x_test)

print evaluate(y_pred_test_csdt, y_test, cost_mat_test)

#cost-sensitive lr
CSLR = CostSensitiveLogisticRegression()
CSLR.fit(x_train, y_train, cost_mat_train)
y_pred_test_cslr = CSLR.predict(x_test)

print evaluate(y_pred_test_cslr, y_test, cost_mat_test)