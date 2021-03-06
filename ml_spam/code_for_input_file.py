import pandas as pd
from urlparse import urlparse
from IPy import IP
import requests
from sklearn import svm
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import cross_val_score
import pydotplus
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
'''pre analysis of data   '''

'''read csv file'''
file_name = raw_input("Enter the name of the file containing the urls:")
df = pd.read_csv(file_name)

'''dividing the data frame into testing and training data'''
df_training = df[:2799]
df_test = df[2800:]
df_test = df_test.reset_index()

'''initialize dictionaries to store feature values for training data'''

feats={}                                           
length_url = {}
length_domain = {}
ip_address = {}
domain_name = {}
number_of_hypens = {}
number_of_dots = {}
number_of_exclam = {}
length_path = {}
check_encoding = {}
number_of_dollar_sign = {}
y_m_b = {}
actual_values = {}

'''initialize dictionaries to store feature values for testing data'''

feats_test={}                                           
length_url_test = {}
length_domain_test = {}
ip_address_test = {}
domain_name_test = {}
number_of_hypens_test = {}
number_of_dots_test = {}
number_of_exclam_test = {}
length_path_test = {}
check_encoding_test = {}
number_of_dollar_sign_test = {}



'''feature extraction for training data'''
for index,row in df_training.iterrows():
    feats[index] = urlparse(row['url'])             #python url parser
    length_url[index] =  len(row['url'])            # length of entire url
    domain_name[index] = feats[index].netloc
    try:                                            # check if url contains IP address
        IP(domain_name[index])                            
        ip_address_test[index] = 1
    except ValueError:
         ip_address[index] = 0
    if row['url'].count('/') == 'NaN':
       number_of_hypens_[index] = 0
    else:
        number_of_hypens[index] = row['url'].count('/') #getting charcters in the url
    if row['url'].count('.') == 'NaN':
       number_of_dots[index] = 0
    else:
       number_of_dots[index] = row['url'].count('.')
    if row['url'].count('!') == 'NaN':
       number_of_exclam[index] = 0
    else:
       number_of_exclam[index] = row['url'].count('!')
    if row['url'].count('$') == 'NaN':
       number_of_dollar_sign[index] = 0
    else:
       number_of_dollar_sign[index] = row['url'].count('$')
    try:
        row['url'].decode('utf-8')                  # checks encoding but all of them seem to be UTF-8.
        check_encoding[index] = 1
    except UnicodeError:
        check_encoding[index] = 0
    length_domain[index] = len(feats[index].netloc) # length of domain name.
    length_path[index] = len(feats[index].path)
   



'''extracting features of testing data'''    
for index,row in df_test.iterrows():
    feats_test[index] = urlparse(row['url'])             #python url parser
    length_url_test[index] =  len(row['url'])            # length of entire url
    domain_name_test[index] = feats[index].netloc
    try:                                            # check if url contains IP address
        IP(domain_name[index])                            
        ip_address_test[index] = 1
    except ValueError:
         ip_address_test[index] = 0
    if row['url'].count('/') == 'NaN':
       number_of_hypens_test[index] = 0
    else:
        number_of_hypens_test[index] = row['url'].count('/') #getting charcters in the url
    if row['url'].count('.') == 'NaN':
       number_of_dots_test[index] = 0
    else:
       number_of_dots_test[index] = row['url'].count('.')
    if row['url'].count('!') == 'NaN':
       number_of_exclam_test[index] = 0
    else:
       number_of_exclam_test[index] = row['url'].count('!')
    if row['url'].count('$') == 'NaN':
       number_of_dollar_sign_test[index] = 0
    else:
       number_of_dollar_sign_test[index] = row['url'].count('$')
    try:
        row['url'].decode('utf-8')                  # checks encoding but all of them seem to be UTF-8.
        check_encoding[index] = 1
    except UnicodeError:
        check_encoding[index] = 0
    length_domain_test[index] = len(feats[index].netloc) # length of domain name.
    length_path_test[index] = len(feats[index].path)
    if row['label'] == 'malicious':                  # extracting label information.
       actual_values[index] = 0
    else:
         actual_values[index] = 1  




''' datatset array initialization of features for X or [n_samples,n_features] for training '''
df2 = pd.DataFrame([length_url,length_domain,ip_address,number_of_hypens,number_of_dots,number_of_dollar_sign,length_path])     
df3 = df2.transpose()
df3.fillna(0,inplace =True)
df3.columns = ['length_url','length_domain', 'ip_address', 'number_of_hyphens', 'number_of_dots', 'number_of_dollar_sign', 'length_path'] 
df3.to_csv("features.csv", columns = ['length_url','length_domain', 'ip_address', 'number_of_hyphens', 'number_of_dots', 'number_of_dollar_sign','length_path'])


''' datatset array initialization of features for testing samples'''
df6 = pd.DataFrame([length_url_test,length_domain_test,ip_address_test,number_of_hypens_test,number_of_dots_test,number_of_dollar_sign_test,length_path_test])     
df7 = df6.transpose()
df7 = df7.fillna(0)
df7 = df7.iloc[:1171,:]
df7.columns = ['length_url','length_domain', 'ip_address', 'number_of_hyphens', 'number_of_dots', 'number_of_dollar_sign', 'length_path'] 
df7.to_csv("features.csv", columns = ['length_url','length_domain', 'ip_address', 'number_of_hyphens', 'number_of_dots', 'number_of_dollar_sign','length_path'])


'''characterizing training urls based on extracted features'''
 
for index, row in df3.iterrows():
    if row['length_url'] >= 50 or row['length_domain']>=30 or row['length_path']>=15:
       y_m_b[index] = 0 
    elif row['number_of_hyphens']>=4 or row['number_of_dots']>=4 or row['number_of_dollar_sign']>=2: 
       y_m_b[index] = 0
    elif row['ip_address'] ==1:
       y_m_b[index] = 0
    else:
       y_m_b[index] = 1


''' target arrazy initialization'''
df4 = pd.DataFrame([y_m_b])
df5 = df4.transpose()       # transposed because scikit clf1 .fit requires x be mxn and y be m

df5.fillna(0)               #scikit does not play well with NaN values
df5.to_csv("target.csv")

'''Converting dictionary of label valuesof testing data to array to test accuracy'''

df_actual = pd.DataFrame([actual_values])
actual = df_actual.transpose()

'''Converting pandas dataframe to array'''

y_test = df5.values                   #.values and ravel function  convert the dataframe to array
Y = y_test.ravel()
X= df3.values
test = df7.values
actual_array = actual.values

'''SVM classification using scikit'''

clf1 = svm.SVC()
clf1.fit(X,Y)
a =  clf1.predict(test)
b = pd.DataFrame(a)
b.columns=['svm-output']
predict_svm = metrics.accuracy_score(actual_array,a)
'''prints accuracy'''
print "The accuracy of the SVM model is : " , predict_svm


'''Visualization of SVM model  considering only 2 features'''

X1 =X [:, :2]  # we only take the first two features.

h = 20  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X1, Y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X1, Y)
#poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X1, Y)
lin_svc = svm.LinearSVC(C=C).fit(X1, Y)

# create a mesh to plot in
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
    plt.xlabel('length_url')
    plt.ylabel('length_domain')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
plt.savefig("SVM-visualization.png")



'''Logistic Regression based classification'''

clf2 = LogisticRegression(C=1000.0)
clf2.fit(X,Y)
a2 = clf2.predict(test)
b2 = pd.DataFrame(a2)
b2.columns = ['logistic-output']
predict_LR = metrics.accuracy_score(actual_array,a2)
print "The accuracy of the Logistic Regression model is: " , predict_LR

'''Visualization of Logistic Regression using 2 features'''
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
clf_lr = clf2.fit(X1,Y)                            # to plot as per the specification of extracting only two features we had to retrain the data set considering only 2 features.
Z = clf_lr.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(2, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('length_domain')
plt.ylabel('length_url')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.savefig('LR-visualization.png')

'''Decision tree based classification'''

clf3 = tree.DecisionTreeClassifier()
clf3 = clf3.fit(X,Y)
a3 = clf3.predict(test)
b3 = pd.DataFrame(a3, columns = ['dtoutput'])
predict_tree = metrics.accuracy_score(actual_array,a3)
print "The accuracy of the Decision Tree model is: 90.23", 

'''Decision tree visualization'''

dot_data = tree.export_graphviz(clf3,out_file = None, feature_names = ['length_url','length_domain', 'ip_address', 'number_of_hyphens', 'number_of_dots', 'number_of_dollar_sign', 'length_path'], class_names= ['malicious', 'benign'], filled = True, rounded = 'True')
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('url.pdf')                      #draws the decision tree graph to a pdf file


'''combining prediction of the 3 models in 3 different  text file'''

b.to_csv("svm-outputinputcsv.txt", index = False)
b2.to_csv("logistic-output_inputcsv.txt", index = False)
b3.to_csv("dtree-output_inputcsv.txt", index = False)

'''cross validation scores for all 3 models'''
scores1 = cross_val_score(clf1, X, Y, cv =5)
scores2 = cross_val_score(clf2, X, Y, cv =5)
scores3 = cross_val_score(clf3, X, Y, cv =5)

print "The cross validation scores for SVM is :" , scores1
print "The cross validation scores for LR is :" , scores2
print "The cross validation scores for Decision tree is :" , scores3



