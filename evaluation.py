import numpy as np
import pandas as pd
from costcla.metrics import savings_score


class evaluation:

    def __init__(self, Amounts):
        self.columns = ["recall", "precision", "specificity", "saving", "C_recall", "C_precision", "C_specificity", "C_saving"]
        self.results = []
        self.amounts = Amounts
        self.cost_matrix = self.cost_matrix(self.amounts, .22)

        self.precision = lambda tp, fp: float(tp)/(tp + fp) if (tp + fp) > 0 else 0
        self.recall = lambda tp, fn: float(tp)/(tp + fn) if (tp + fn) > 0 else 0
        self.specificity = lambda tn, fp: float(tn)/(tn+fp) if (tn+fp) > 0 else 0

    def cost_matrix(self, Amounts, Ca):
        cmatrix = np.ones((Amounts.shape[0], 4))

        cmatrix[:, 0] *= Ca
        cmatrix[:, 1] = Amounts
        cmatrix[:, 2] *= Ca
        cmatrix[:, 3] *= 0

        return cmatrix

    def evaluate(self, y_, y):

        tp, tn, fp, fn = 0, 0, 0, 0

        for prediction, label in zip(np.argmax(y_, axis=1), np.argmax(y, axis=1)):

            if label == 1 and prediction == 1:
                tp += 1
            elif label == 0 and  prediction == 0:
                tn += 1
            elif label == 1 and prediction == 0:
                fn += 1
            else:
                fp += 1

        r = self.recall(tp, fn)
        p = self.precision(tp, fp)
        s = self.specificity(tn, fp)

        savings = savings_score(np.argmax(y, axis=1), np.argmax(y_, axis=1), self.cost_matrix)

        C_recall, C_precision, C_specificity, C_savings = self.cost_sensitive(y_, y, self.amounts)


        #print n_samples, r, p, f1, savings, C_recall, C_precision, C_F1, C_savings

        self.results.append([r, p, s, savings, C_recall, C_precision, C_specificity, C_savings])

    def cost_sensitive(self, y_, y, Amounts):

        costmatrix = np.zeros((y.shape[0], 2))
        costmatrix[:, 0] = np.ones(y.shape[0]) * 0.22
        costmatrix[:, 1] = Amounts

        thresholds = costmatrix[:, 0] / costmatrix[:, 1]

        pred = [1 if y_i > threshold else 0 for y_i, threshold in zip(y_[:, 1], thresholds)]
        pred = np.array(pred)

        tp, tn, fp, fn = 0, 0, 0, 0

        for prediction, label in zip(pred, np.argmax(y, axis=1)):

            if label == 1 and prediction == 1:
                tp += 1
            elif label == 0 and  prediction == 0:
                tn += 1
            elif label == 1 and prediction == 0:
                fn += 1
            else:
                fp += 1

        r = self.recall(tp, fn)
        p = self.precision(tp, fp)
        s = self.specificity(tn, fp)

        savings = savings_score(np.argmax(y, axis=1), pred, self.cost_matrix)

        return r, p, s, savings

    def get_results(self):
        return pd.DataFrame(data=self.results, columns=self.columns)

    def get_average_scores(self):
        self.results.append(list(np.mean(self.results, axis=0)))
        return pd.DataFrame(data=self.results, columns=self.columns)



