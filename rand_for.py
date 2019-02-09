import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from numpy import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

'''reading the csv file with feature vectors and labels'''

df = pd.read_csv("labelled_data.csv")


# removing redundant columns

df = df.drop("Unnamed: 0",1)
df = df.drop("Unnamed: 0.1", 1)

''' converting pandas dataframe to numpy array'''

X = array(df.ix[:,1:13])   # feature vectors
Y = array(df.ix[:,14:15])  # labels

# selecting random data for training and testing

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 6)

RANDOM_STATE = 6



ensemble_clfs = [

    ("RandomForestClassifier, max_depth='2'",

        RandomForestClassifier(warm_start=True, oob_score=True,

                               max_features="sqrt",
			       max_depth = 2,
                               
                               random_state=RANDOM_STATE)),

    ("RandomForestClassifier, max_features='3'",

        RandomForestClassifier(warm_start=True, max_features='log2', 
                               max_depth = 4,

                               oob_score=True,

                               random_state=RANDOM_STATE))
]




error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

min_estimators = 15

max_estimators = 175



for label, clf in ensemble_clfs:

    for i in range(min_estimators, max_estimators + 1):

        clf.set_params(n_estimators=i)

        clf.fit(X_train, Y_train)
    


        # Record the OOB error for each `n_estimators=i` setting.

        oob_error = 1 - clf.oob_score_

        error_rate[label].append((i, oob_error))

    print oob_error

# Generate the "OOB error rate" vs. "n_estimators" plot.

for label, clf_err in error_rate.items():

    xs, ys = zip(*clf_err)

    plt.plot(xs, ys, label=label)
for label,clf in ensemble_clfs:
    y_hat = clf.predict(X_test)
    precision = precision_score(Y_test, y_hat, average = 'weighted')
    recall = recall_score(Y_test, y_hat, average = 'weighted')
    print recall
    print precision

plt.xlim(min_estimators, max_estimators)

plt.xlabel("n_estimators")

plt.ylabel("OOB error rate")

plt.legend(loc="upper right")

plt.show()

