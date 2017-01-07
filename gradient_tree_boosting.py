import csv

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

x_data_train = pd.read_csv("data/X_train.csv", header=None, delimiter=",", dtype='float32').values
y_data_train = pd.read_csv("data/y_train.csv", header=None, delimiter=",", dtype='float32').values.ravel()
x_data_valid = pd.read_csv("data/X_valid.csv", header=None, delimiter=",", dtype='float32').values
y_data_valid = pd.read_csv("data/y_valid.csv", header=None, delimiter=",", dtype='float32').values.ravel()


clf = Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(C=0.01, penalty="l1", dual=False))),
    ('classification', GradientBoostingClassifier(n_estimators=1000, learning_rate=0.06, max_depth=13, verbose=1,
                                 max_features='sqrt', min_samples_leaf=30, min_samples_split=400))
])
clf.fit(x_data_train, y_data_train)
# y_predicted = clf.predict(x_data_valid)

x_data_test = pd.read_csv("data/X_test.csv", header=None, delimiter=",", dtype='float32').values
y_predicted = clf.predict(x_data_test)
res = list(zip(range(1, len(y_predicted)+1), y_predicted))
csvfile = "out/res.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(res)
