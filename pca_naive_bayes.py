
from __future__ import print_function

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from numpy import *

import pandas as pd

from sklearn.pipeline import make_pipeline

FIG_SIZE = (10, 7)

df = pd.read_csv("labelled_data.csv")


# removing redundant columns

df = df.drop("Unnamed: 0",1)
df = df.drop("Unnamed: 0.1", 1)

''' converting pandas dataframe to numpy array'''

X = array(df.ix[:,1:13])   # feature vectors
Y = array(df.ix[:,14:15])  # labels




X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 6)

RANDOM_STATE = 6


# Fit to data and predict using pipelined GNB and PCA.

unscaled_clf = make_pipeline(PCA(n_components=5), GaussianNB())

unscaled_clf.fit(X_train, y_train)

pred_test = unscaled_clf.predict(X_test)



# Fit to data and predict using pipelined scaling, GNB and PCA.

std_clf = make_pipeline(StandardScaler(), PCA(n_components=5), GaussianNB())

std_clf.fit(X_train, y_train)

pred_test_std = std_clf.predict(X_test)



# Show prediction accuracies in scaled and unscaled data.

print('\nPrediction accuracy for the normal test dataset with PCA')

print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))



print('\nPrediction accuracy for the standardized test dataset with PCA')

print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))



# Extract PCA from pipeline

pca = unscaled_clf.named_steps['pca']

pca_std = std_clf.named_steps['pca']





scaler = std_clf.named_steps['standardscaler']

X_train_std = pca_std.transform(scaler.transform(X_train))



# visualize standardized vs. untouched dataset with PCA performed

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)




