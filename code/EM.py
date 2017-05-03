#======================================================================================
# Description: This code applies the EM algorithm using multiple classifiers
# Dependencies: pandas, numpy, sklearn, operator, scipy
#======================================================================================

# coding=utf-8
# Import libraries
import pandas as pd
import numpy as np
import sklearn
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import operator

# Function to split into two datasets
def semi_supervised_split(train, size_test):
    train_labeled, train_unlabeled = train_test_split(train, test_size=size_test)
    return train_labeled, train_unlabeled

# This functions holds all classifiers we are using and calls the prediction function to apply
# the EM algorithm. Uncomment out sections as needed and run depending on the classfier
# output you desire.
def function_loop(sizes, train, test, label_nam, features):
    
    clf1 = GaussianNB()
    clf2 = GaussianNB()
    clf3 = GaussianNB()
    print "Naive Bayes"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    '''
    clf1 = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)
    clf2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)
    clf3 = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)
    print "Logistic Regression with Multinomial lbfgs"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)

    clf1 = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=200)
    clf2 = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=200)
    clf3 = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=200)
    print "Logistic Regression with Multinomial newton-cg"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)

    clf1 = LogisticRegression(multi_class="multinomial", solver="sag", max_iter=200)
    clf2 = LogisticRegression(multi_class="multinomial", solver="sag", max_iter=200)
    clf3 = LogisticRegression(multi_class="multinomial", solver="sag", max_iter=200)
    print "Logistic Regression with Multinomial sag"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    
    clf1 = LinearSVC()
    clf2 = LinearSVC()
    clf3 = LinearSVC()
    print "LinearSVC"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    
    clf1 = SVC(kernel="poly", probability=True)
    clf2 = SVC(kernel="poly", probability=True)
    clf3 = SVC(kernel="poly", probability=True)
    print "SVC with poly"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    
    clf1 = SVC(kernel="rbf", probability=True)
    clf2 = SVC(kernel="rbf", probability=True)
    clf3 = SVC(kernel="rbf", probability=True)
    print "SVC with rbf"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    
    clf1 = SVC(kernel="sigmoid", probability=True)
    clf2 = SVC(kernel="sigmoid", probability=True)
    clf3 = SVC(kernel="sigmoid", probability=True)
    print "SVC with sigmoid"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    
    clf1 = KNeighborsClassifier(weights="uniform", algorithm="ball_tree", n_neighbors=n)
    clf2 = KNeighborsClassifier(weights="uniform", algorithm="ball_tree", n_neighbors=n)
    clf3 = KNeighborsClassifier(weights="uniform", algorithm="ball_tree", n_neighbors=n)
    print "KNN with uniform and ball_tree"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    
    clf1 = KNeighborsClassifier(weights="uniform", algorithm="kd_tree", n_neighbors=n)
    clf2 = KNeighborsClassifier(weights="uniform", algorithm="kd_tree", n_neighbors=n)
    clf3 = KNeighborsClassifier(weights="uniform", algorithm="kd_tree", n_neighbors=n)
    print "KNN with uniform and kd_tree"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    
    clf1 = KNeighborsClassifier(weights="distance", algorithm="ball_tree", n_neighbors=n)
    clf2 = KNeighborsClassifier(weights="distance", algorithm="ball_tree", n_neighbors=n)
    clf3 = KNeighborsClassifier(weights="distance", algorithm="ball_tree", n_neighbors=n)
    print "KNN with distance and ball_tree"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    
    clf1 = KNeighborsClassifier(weights="distance", algorithm="kd_tree", n_neighbors=n)
    clf2 = KNeighborsClassifier(weights="distance", algorithm="kd_tree", n_neighbors=n)
    clf3 = KNeighborsClassifier(weights="distance", algorithm="kd_tree", n_neighbors=n)
    print "KNN with distance and kd_tree"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    
    clf1 = DecisionTreeClassifier()
    clf2 = DecisionTreeClassifier()
    clf3 = DecisionTreeClassifier()
    print "Decision Tree"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)

    clf1 = RandomForestClassifier()
    clf2 = RandomForestClassifier()
    clf3 = RandomForestClassifier()
    print "Random Forest"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    
    clf1 = GradientBoostingClassifier()
    clf2 = GradientBoostingClassifier()
    clf3 = GradientBoostingClassifier()
    print "Gradient Boosting"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)

    clf1 = AdaBoostClassifier()
    clf2 = AdaBoostClassifier()
    clf3 = AdaBoostClassifier()
    print "Ada Boost Classifier"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    
    clf1 = MLPClassifier(activation="identity")
    clf2 = MLPClassifier(activation="identity")
    clf3 = MLPClassifier(activation="identity")
    print "MLP Neural Net with identity"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)

    clf1 = MLPClassifier(activation="logistic")
    clf2 = MLPClassifier(activation="logistic")
    clf3 = MLPClassifier(activation="logistic")
    print "MLP Neural Net with logistic"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)

    clf1 = MLPClassifier(activation="tanh")
    clf2 = MLPClassifier(activation="tanh")
    clf3 = MLPClassifier(activation="tanh")
    print "MLP Neural Net with tanh"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)

    clf1 = MLPClassifier(activation="relu")
    clf2 = MLPClassifier(activation="relu")
    clf3 = MLPClassifier(activation="relu")
    print "MLP Neural Net with relu"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    
    clf1 = linear_model.SGDClassifier(loss="log")
    clf2 = linear_model.SGDClassifier(loss="log")
    clf3 = linear_model.SGDClassifier(loss="log")
    print "Stochastic Gradient Classifier"
    prediction_func(clf1, clf3, sizes, train, test, label_nam, features)
    '''

# Prediction algorithm to apply co-training algorithm using specified classifier
def prediction_func(clf1, clf3, sizes, train, test, label_name1, features):
    x = pd.Series()
    
    # For different proportions of labeled and unlabeled training data
    for size_test in sizes:
        print "Unlabeled Training Data Size: ", size_test
        train_labeled, train_unlabeled = semi_supervised_split(train, size_test)
        count = 0
        accuracy_list = []
        precision_list = []
        recall_list = []
        fscore_list = []
        testing_data = train_unlabeled.ix[:, :-1]
        testing_labels = train_unlabeled.ix[:, -1]
        val1 = 0
        val2 = 1
        
        # Check if there is an increase in correct predictions OR the test (unlabeled training) dataset is empty
        while val1 != val2 and len(testing_data) != 0:
            test_check = pd.DataFrame(columns=list(testing_data))
            label_check = []

            val2 = val1
            
            # Train with labeled data and predict using unlabeled data using classifier model
            clf1.fit(train_labeled.ix[:, :-1], train_labeled.ix[:, -1])
            prediction1 = clf1.predict(testing_data)
            prediction1 = pd.DataFrame(prediction1)
            prediction1.columns = [label_name1]

            testing_data.reset_index(drop=True, inplace=True)
            prediction1.reset_index(drop=True, inplace=True)
            temp1 = pd.concat([testing_data, prediction1], axis=1)
            temp1.reset_index(drop=True, inplace=True)

            testing_labels.reset_index(drop=True, inplace=True)
            testing_labels.columns = [label_name1]
            k = 0
            
            # Keep predicting the labeled data and adding to unlabeled data
            for i in range(len(temp1)):
                if temp1.ix[i, label_name1] == testing_labels.ix[i, label_name1]:
                    temp1.reset_index(drop=True, inplace=True)
                    train_labeled.reset_index(drop=True, inplace=True)
                    train_labeled = train_labeled.append(temp1.ix[i], ignore_index=True)
                else:
                    temp1.reset_index(drop=True, inplace=True)
                    test_check.reset_index(drop=True, inplace=True)
                    x = temp1.ix[i, :-1]
                    y = pd.DataFrame(x.reshape((1, len(features) - 1)))
                    y.columns = [list(testing_data)]
                    y.reset_index(drop=True, inplace=True)
                    test_check = test_check.append(y, ignore_index=True)
                    label_check.append(testing_labels.ix[i, label_name1])

            testing_data = test_check
            testing_labels = pd.DataFrame(label_check)

            val1 = len(train_labeled)

            # Train another classifier model using updated training labeled data and predict accuracy using test data
            clf3.fit(train_labeled.ix[:, :-1], train_labeled.ix[:, -1])
            prediction = clf3.predict(test.ix[:, :-1])
            prediction = pd.DataFrame(prediction)
            prediction.columns = [label_name1]
            score = accuracy_score(test.ix[:, -1], prediction)
            precision = precision_score(test.ix[:, -1], prediction, average='macro')
            recall = recall_score(test.ix[:, -1], prediction, average='macro')
            f1 = f1_score(test.ix[:, -1], prediction, average='macro')
            accuracy_list.append(score)
            precision_list.append(precision)
            recall_list.append(recall)
            fscore_list.append(f1)
            count = count + 1
        max_index, max_accuracy = max(enumerate(accuracy_list), key=operator.itemgetter(1))
        max_precision = precision_list[max_index]
        max_recall = recall_list[max_index]
        max_f1 = fscore_list[max_index]
        print "Max accuracy: ", max_accuracy
        print "Max precision: ", max_precision
        print "Max recall: ", max_recall
        print "Max f1 score: ", max_f1

# Main function to call functions to apply co-training algorithm
if __name__ == '__main__':
    wine_data = pd.read_csv("dataset/winequality-white.csv", sep=";")
    waveform_data = pd.read_csv("dataset/waveform-+noise.data.csv", sep=";")
    sensor_data = pd.read_csv("dataset/sensor_readings_24.data.csv", sep=";")
    test_size = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

    n = len(set(wine_data.iloc[:, -1]))
    feature_set = ["fixed acidity", "volatile acidity", "residual sugar", "total sulfur dioxide", "sulphates", "alcohol", "citric acid", "chlorides", "free sulfur dioxide", "density", "pH", "quality"]
    train_data, test_data = train_test_split(wine_data, test_size=0.3)
    label_name = "quality"
    print
    print "WINE DATASET"
    function_loop(test_size, train_data, test_data, label_name, feature_set)

    n = len(set(sensor_data.iloc[:, -1]))
    feature_set = ["f2", "f4", "f6", "f7", "f10", "f14", "f18", "f22", "f24", "f1", "f3", "f5", "f8", "f9", "f11", "f12", "f13", "f15", "f16", "f17", "f19", "f20", "f21", "f23", "class"]
    train_data, test_data = train_test_split(sensor_data, test_size=0.3)
    label_name = "class"
    print
    print "SENSOR DATASET"
    function_loop(test_size, train_data, test_data, label_name, feature_set)

    n = len(set(waveform_data.iloc[:, -1]))
    feature_set = ["f1", "f2", "f3", "f4", "f6", "f10", "f12", "f13", "f14", "f15", "f18", "f20", "f21", "f22", "f24", "f5", "f7", "f8", "f9", "f11", "f16", "f17", "f19", "f23", "f25", "f27", "f29",
                   "f30", "f31", "f33", "f34", "f38", "f39", "f26", "f28", "f32", "f35", "f36", "f37", "f40", "class"]
    train_data, test_data = train_test_split(waveform_data, test_size=0.3)
    label_name = "class"
    print
    print "WAVEFORM DATASET"
    function_loop(test_size, train_data, test_data, label_name, feature_set)
