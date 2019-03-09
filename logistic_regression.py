from random import seed
from random import randrange
from csv import reader
from math import exp
import numpy as np

seed(2)

# read csv file
def read_csv(filename):
    data = list()
    with open(filename, 'rU') as file:
         csv_reader = reader(file)
         for row in csv_reader:
             if not row:
                continue
             data.append(row)
    return data


# convert string to float
def str_to_float(data,column):
     for row in data:
         row[column] = float(row[column].strip())

def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


'''
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
'''

def accuracy_metric(actual, predicted):
	actual1 = np.asarray(actual)
	predicted1 = np.asarray(predicted)
	mse = np.mean((actual1 - predicted1)**2)
	return mse

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))


# stochastic gradient descent with early stopping
def coefficients_sgd(train, l_rate, n_epoch, early_stop_set, coef_selector):
	if coef_selector == "1":
		coef = [0.0 for i in range(len(train[0]))]
	else:
		coef =[0.64425535, -0.48418462,  0.01805113, -0.53775428,  0.58358285,  0.74945579,
 -0.50760247, -0.07090949,  0.26018304, -0.61116985, -1.50504523,  0.371147,
  1.17764306, -0.54653043, -1.73298373,  0.82829164,  0.8217108,  -0.84495329,
 -0.33792605,  1.65046924, -0.0586316,   0.08735047, -1.18899314, -0.28959529,
 -0.17994259, -0.97480614,  0.13426974,  0.10069284, -1.07248982, -0.91572244,
 -0.70169577,  0.47521413, -1.29088629, -0.29417533, -0.49213544,  1.23959051,
  0.17613014,  0.44853473,  0.07496574,  0.7353836,  -0.82565314, -1.37954018,
 -0.43492872, -0.04660075,  0.09009324,  0.31359551,  0.34304235, -0.47520017,
 -0.49590361,  0.0857984]
	print coef
	count = 0
	temp = 100
	for epoch in range(n_epoch):
		if count > 4:
			break
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
			error_early = []
			for row in early_stop_set:
				yhat_early = predict(row,coef)
				error_early.append(row[-1] - yhat_early)
			if (sum(error_early)/len(error_early)) > temp:
					count += 1
			else:
					count = 0
		temp = sum(error_early)/len(error_early)
	return coef, error_early
 
# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(test, coef, l_rate, n_epoch):
	predictions = list()
#	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
	return(predictions)

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, coef_selector, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		early_stop_set = train_set[:50]
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		coef, error_early = coefficients_sgd(train_set, l_rate, n_epoch, early_stop_set, coef_selector)
		predicted = algorithm(test_set, coef,*args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores, error_early


filename = 'dataset.csv'
data = read_csv(filename)
data.remove(data[0])
for i in range(len(data[0])):
		str_to_float(data,i)

data_split = cross_validation_split(data, 5)


n_folds = 5 # 5 folds as mentioned in the problem
l_rate = raw_input("Enter learning rate") # learning rate
l_rate = float(l_rate)
coef_selector = input("What initial weights configuration do you desire: \n1. 0 \n2. random with uniform distributin.")
n_epoch = 100 # real experiments generally assign a value ~1 mil

scores, error_early = evaluate_algorithm(data, logistic_regression, n_folds, coef_selector,l_rate, n_epoch)
#print('error_early: %s' % error_early)
print('MSE: %s' % scores)
print('Mean MSE: %.3f' % (sum(scores)/float(len(scores))))
