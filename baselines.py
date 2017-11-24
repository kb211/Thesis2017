from sklearn.ensemble import RandomForestClassifier
from costcla.models import CostSensitiveDecisionTreeClassifier
from costcla.metrics import savings_score
from evaluation import evaluation
import math
import numpy as np
import pandas as pd



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


X, Y, ids, Amounts = preprocess()
ratio = int(math.ceil(.8 * len(X)))
x_train, y_train, x_test, y_test = X[:ratio], Y[:ratio], X[ratio:], Y[ratio:]

evaluation_csdt = evaluation(Amounts[ratio:])
evaluation_rf = evaluation(Amounts[ratio:])

# cost_mat [false positives, false negatives, true positives, true negatives]
cost_mat = np.ones((X.shape[0], 4))
cost_mat[:, 0] *= 0.22
cost_mat[:, 1] = Amounts
cost_mat[:, 2] *= 0.22
cost_mat[:, 3] *= 0

cost_mat_train, cost_mat_test = cost_mat[:ratio], cost_mat[ratio:]

rfc = RandomForestClassifier(random_state=0)
y_pred_test_rf = rfc.fit(x_train, np.argmax(y_train, axis=1)).predict_proba(x_test)



f = CostSensitiveDecisionTreeClassifier()
y_pred_test_csdt = f.fit(x_train, np.argmax(y_train, axis=1), cost_mat_train).predict_proba(x_test)


print np.argmax(y_test, axis=1).shape, y_pred_test_rf.shape, y_pred_test_csdt .shape

evaluation_csdt.evaluate(y_pred_test_csdt, y_test)
evaluation_rf.evaluate(y_pred_test_rf, y_test)

print evaluation_rf.get_results()
print evaluation_csdt.get_results()

#Savings using CSDecisionTree
print(savings_score(np.argmax(y_test, axis=1), f.predict(x_test), cost_mat_test))

#Savings using only RandomForest
print(savings_score(np.argmax(y_test, axis=1), rfc.predict(x_test), cost_mat_test))
