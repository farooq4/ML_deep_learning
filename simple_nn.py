import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

beta = 0.1
graph = tf.Graph()

no_of_hidden_layers = int(raw_input("Enter the desired number of hidden layers: \n"))
no_of_neurons = int(raw_input("Enter the number of hidden neurons: "))



#reading data into pandas dataframe and seperating features and labels
df_train = pd.read_csv("spambase_mod.csv")

X_train_df = df_train.iloc[:,:-1]

Y_train_df = df_train.iloc[:,-1]



#normalizing features and converting to numpy array to feed to the neural network
X_train_arr = X_train_df.values

min_max_scaler = preprocessing.MinMaxScaler()			# used to compute value with zero mean and unit variance

X_train = min_max_scaler.fit_transform(X_train_arr)

Y_train = Y_train_df.values					# converting label dataframe to numpy array



#split arrays into test and train set
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.13)



#one hot encoding of labels (not necessary since binary class but needed for multiclass classification)
labels_train = (np.arange(2) == Y_train[:,None]).astype(np.float32)

labels_test = (np.arange(2) == Y_test[:,None]).astype(np.float32)



#assigning features and labels to placeholder structs
inputs = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name = 'inputs')

label =  tf.placeholder(tf.float32, shape=(None,2), name = 'labels') 



# initializing neural network hidden and output  layers

# First layer
hid1_size = no_of_neurons
hid2_size = no_of_neurons

w1 = tf.Variable(tf.random_normal([hid1_size, X_train.shape[1]], stddev=0.01), name='w1')

b1 = tf.Variable(tf.constant(0.1, shape=(hid1_size, 1)), name='b1')

y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(inputs)), b1)), keep_prob=0.5)

if no_of_hidden_layers == 2:

				# Second layer
				

				w2 = tf.Variable(tf.random_normal([hid2_size, hid1_size], stddev=0.01), name='w2')

				b2 = tf.Variable(tf.constant(0.1, shape=(hid2_size, 1)), name='b2')

				y2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w2, y1), b2)), keep_prob=0.5)



				# Output layer
 
				wo = tf.Variable(tf.random_normal([2, hid2_size], stddev=0.01), name='wo')

				bo = tf.Variable(tf.random_normal([2, 1]), name='bo')

				yo = tf.transpose(tf.add(tf.matmul(wo, y2), bo))

elif no_of_hidden_layers == 1:
				# Output layer
 
				wo = tf.Variable(tf.random_normal([2, hid2_size], stddev=0.01), name='wo')

				bo = tf.Variable(tf.random_normal([2, 1]), name='bo')

				yo = tf.transpose(tf.add(tf.matmul(wo, y1), bo))



#loss function, regulairzation and back propogation
lr = tf.placeholder(tf.float32, shape=(), name = 'learning_rate')

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yo, labels = label))	#original loss fucntion

if no_of_hidden_layers == 2:
				regularizer = tf.nn.l2_loss(wo) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w1)
elif no_of_hidden_layers == 1:
				regularizer = tf.nn.l2_loss(wo) + tf.nn.l2_loss(w1)

loss = tf.reduce_mean(loss + beta*regularizer)

optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)


 

#prediction of output values
pred = tf.nn.softmax(yo)

pred_label = tf.argmax(pred,1)

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(label,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



#initialzing session
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)


for learning_rate in [0.1]:
    for epoch in range(25):
        avg_cost = 0.0

        # For each epoch, we go through all the samples we have.
        for i in range(X_train.shape[0]):
            # Finally, this is where the magic happens: run our optimizer, feed the current example into X and the current target into Y
            _, c = sess.run([optimizer, loss], feed_dict={lr:learning_rate, 
                                                          inputs: X_train[i, None],
                                                          label: labels_train[i, None]})


'''
acc_train = accuracy.eval(feed_dict={inputs: X_train, label: labels_train})
print("Train accuracy: {:3.2f}%".format(acc_train*100.0))
'''

acc_test = accuracy.eval(feed_dict={inputs: X_test, label: labels_test})
print("Test accuracy:  {:3.2f}%".format(acc_test*100.0))

