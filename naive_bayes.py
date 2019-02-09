from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics
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

# scaling data for naive bayes classification(naive bayes assumes gaussian distribution)
std_scale = preprocessing.StandardScaler().fit(X_train)

X_train_std = std_scale.transform(X_train)

X_test_std = std_scale.transform(X_test)


# naive bayes classifier
clf_std = GaussianNB()

fit_std = clf_std.fit(X_train_std, Y_train)

#pred_train_std = clf_std.predict(X_train_std)

pred_test_std = clf_std.predict(X_test_std)


recall = recall_score(Y_test, pred_test_std, average = 'weighted')
precision = precision_score(Y_test, pred_test_std, average = 'weighted')


clf = GaussianNB()
fit = clf.fit(X_train, Y_train)
pred_test = clf.predict(X_test)

recall_un = recall_score(Y_test, pred_test, average = 'weighted')
precision_un = precision_score(Y_test, pred_test, average = 'weighted')




print "Recall for Naive Bayes with feature vector normalized is: "
print recall
print "Precision for Naive Bayes with feature vector normalized is: "
print precision




print "Recall for Naive Bayes with feature vector not normalized is: "
print recall_un
print "Precision for Naive Bayes with feature vector not normalized is: "
print precision_un

