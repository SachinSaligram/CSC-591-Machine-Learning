import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#Artificial Neural Network implemeted using tensorflow
def AnnSingleLayer(features_train, features_tets, labels_train, labels_test):
    #getting the dimension of the training set, number of features
    n_dim = features_train.shape[1]
    
    #reshaping train and data test labels
    labels_train = np.reshape(labels_train,[labels_train.shape[0],1])
    labels_test = np.reshape(labels_test,[labels_test.shape[0],1])
    
    #standardizing the train and test data
    features_train = (features_train-np.mean(features_train,axis=0))/np.std(features_train,axis=0)
    features_tets = (features_tets-np.mean(features_tets,axis=0))/np.std(features_tets,axis=0)

    #setting the learning rate and the number of epochs (loops)
    learning_rate = 0.01
    training_epochs = 1000

    #setting variables for the formula Y = W*X + b
    X = tf.placeholder(tf.float32,[None,n_dim])
    Y = tf.placeholder(tf.float32,[None,1])
    W = tf.Variable(tf.zeros([n_dim,1]))
    b = tf.Variable(tf.zeros([1]))

    #initializing global variables of tensorflow
    init = tf.global_variables_initializer()

    #setting the formula
    y_ = tf.matmul(X, W)+b
    #setting the cost function
    cost = tf.reduce_mean(tf.square(y_ - Y))
    #defining the training step using gradient descent optimizer
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    #initiating sessions
    sess = tf.Session()
    sess.run(init)

    #running the sessions multiple times (the number of epochs to be exact)
    for epoch in range(training_epochs):
        sess.run(training_step,feed_dict={X:features_train,Y:labels_train})

    #generating predictions, since they are in float converting them to integers
    pred_y1 = np.ceil(sess.run(y_, feed_dict={X: features_tets}))
    pred_y2 = np.floor(sess.run(y_, feed_dict={X: features_tets}))

    #predicting the accuracy, F1 score, precision and recall
    print "Accuracy: %f" % (sum([1 for i in range(len(pred_y1)) if pred_y1[i]==labels_test[i] or pred_y2[i]==labels_test[i]])/float(len(pred_y1)))
    f1_score_val = f1_score(labels_test, pred_y1, average='macro')
    f1_score_val += f1_score(labels_test, pred_y2, average='macro')
    print "F1 score: %f" % f1_score_val
    precision_score_val = precision_score(labels_test, pred_y1, average='macro')
    precision_score_val += precision_score(labels_test, pred_y2, average='macro')
    print "Precision score: %f" % precision_score_val
    recall_score_val = recall_score(labels_test, pred_y1, average='macro')
    recall_score_val += recall_score(labels_test, pred_y2, average='macro')
    print "Recall score: %f" % recall_score_val