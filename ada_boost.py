import time
start = time.time
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from numpy import *
from sklearn import ensemble
from sklearn.model_selection import train_test_split
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



bdt_real = AdaBoostClassifier(

    DecisionTreeClassifier(max_depth=2),

    n_estimators=600,

    learning_rate=1)



bdt_discrete = AdaBoostClassifier(

    DecisionTreeClassifier(max_depth=2),

    n_estimators=600,

    learning_rate=1.5,

    algorithm="SAMME")





bdt_real.fit(X_train, Y_train)

bdt_discrete.fit(X_train, Y_train)



y_hat_real = bdt_real.predict(X_test)
y_hat_discrete = bdt_discrete.predict(X_test)

recall = recall_score(Y_test,y_hat_real, average = 'weighted')
precision = precision_score(Y_test, y_hat_real, average ='weighted')


recall1 = recall_score(Y_test,y_hat_discrete, average = 'weighted')
precision1 = precision_score(Y_test, y_hat_discrete, average ='weighted')

print "recall for real ada is:"
print recall
print "precision for real ada is:"
print precision

print "recall for discrete ada is:"
print recall1
print "precision for discrete ada is:"
print precision1

