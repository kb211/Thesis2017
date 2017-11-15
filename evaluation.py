import numpy as np
import pandas as pd

class evaluation:

    def __init__(self):
        self.columns = ["n_samples", "recall", "precision"]
        self.results = []
        self.predictions = []

    def evaluate(self, y_, y, n_samples):

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

        recall = float(tp)/(tp+fn)
        precision = float(tp)/(tp + fp)
        print precision, recall
        self.results.append([n_samples, recall, precision])

    def get_results(self):
        return pd.DataFrame(data=self.results, columns=self.columns)


