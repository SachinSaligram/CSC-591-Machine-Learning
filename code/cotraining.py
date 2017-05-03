#======================================================================================
# Description: This code applies the co-training algorithm using multiple classifiers
# Dependencies: pandas, numpy, sklearn, operator
#======================================================================================

# coding=utf-8
# Import libraries
import pandas as pd
import numpy as np
import sklearn
import math
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
# the cotraining algorithm. Uncomment out sections as needed and run depending on the classfier
# output you desire.
def function_loop(feature1, feature2, sizes, train, test, label_nam):
    
    clf1 = GaussianNB()
    clf2 = GaussianNB()
    clf3 = GaussianNB()
    print "Naive Bayes"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    '''
    clf1 = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)
    clf2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)
    clf3 = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)
    print "Logistic Regression with Multinomial lbfgs"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)

    clf1 = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=200)
    clf2 = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=200)
    clf3 = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=200)
    print "Logistic Regression with Multinomial newton-cg"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)

    clf1 = LogisticRegression(multi_class="multinomial", solver="sag", max_iter=200)
    clf2 = LogisticRegression(multi_class="multinomial", solver="sag", max_iter=200)
    clf3 = LogisticRegression(multi_class="multinomial", solver="sag", max_iter=200)
    print "Logistic Regression with Multinomial sag"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = LinearSVC()
    clf2 = LinearSVC()
    clf3 = LinearSVC()
    print "LinearSVC"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = SVC(kernel="poly", probability=True)
    clf2 = SVC(kernel="poly", probability=True)
    clf3 = SVC(kernel="poly", probability=True)
    print "SVC with poly"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = SVC(kernel="rbf", probability=True)
    clf2 = SVC(kernel="rbf", probability=True)
    clf3 = SVC(kernel="rbf", probability=True)
    print "SVC with rbf"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = SVC(kernel="sigmoid", probability=True)
    clf2 = SVC(kernel="sigmoid", probability=True)
    clf3 = SVC(kernel="sigmoid", probability=True)
    print "SVC with sigmoid"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = KNeighborsClassifier(weights="uniform", algorithm="ball_tree", n_neighbors=n)
    clf2 = KNeighborsClassifier(weights="uniform", algorithm="ball_tree", n_neighbors=n)
    clf3 = KNeighborsClassifier(weights="uniform", algorithm="ball_tree", n_neighbors=n)
    print "KNN with uniform and ball_tree"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = KNeighborsClassifier(weights="uniform", algorithm="kd_tree", n_neighbors=n)
    clf2 = KNeighborsClassifier(weights="uniform", algorithm="kd_tree", n_neighbors=n)
    clf3 = KNeighborsClassifier(weights="uniform", algorithm="kd_tree", n_neighbors=n)
    print "KNN with uniform and kd_tree"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = KNeighborsClassifier(weights="distance", algorithm="ball_tree", n_neighbors=n)
    clf2 = KNeighborsClassifier(weights="distance", algorithm="ball_tree", n_neighbors=n)
    clf3 = KNeighborsClassifier(weights="distance", algorithm="ball_tree", n_neighbors=n)
    print "KNN with distance and ball_tree"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = KNeighborsClassifier(weights="distance", algorithm="kd_tree", n_neighbors=n)
    clf2 = KNeighborsClassifier(weights="distance", algorithm="kd_tree", n_neighbors=n)
    clf3 = KNeighborsClassifier(weights="distance", algorithm="kd_tree", n_neighbors=n)
    print "KNN with distance and kd_tree"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = DecisionTreeClassifier()
    clf2 = DecisionTreeClassifier()
    clf3 = DecisionTreeClassifier()
    print "Decision Tree"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)

    clf1 = RandomForestClassifier()
    clf2 = RandomForestClassifier()
    clf3 = RandomForestClassifier()
    print "Random Forest"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = GradientBoostingClassifier()
    clf2 = GradientBoostingClassifier()
    clf3 = GradientBoostingClassifier()
    print "Gradient Boosting"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)

    clf1 = AdaBoostClassifier()
    clf2 = AdaBoostClassifier()
    clf3 = AdaBoostClassifier()
    print "Ada Boost Classifier"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = MLPClassifier(activation="identity")
    clf2 = MLPClassifier(activation="identity")
    clf3 = MLPClassifier(activation="identity")
    print "MLP Neural Net with identity"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)

    clf1 = MLPClassifier(activation="logistic")
    clf2 = MLPClassifier(activation="logistic")
    clf3 = MLPClassifier(activation="logistic")
    print "MLP Neural Net with logistic"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)

    clf1 = MLPClassifier(activation="tanh")
    clf2 = MLPClassifier(activation="tanh")
    clf3 = MLPClassifier(activation="tanh")
    print "MLP Neural Net with tanh"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = MLPClassifier(activation="relu")
    clf2 = MLPClassifier(activation="relu")
    clf3 = MLPClassifier(activation="relu")
    print "MLP Neural Net with relu"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    
    clf1 = linear_model.SGDClassifier(loss="log")
    clf2 = linear_model.SGDClassifier(loss="log")
    clf3 = linear_model.SGDClassifier(loss="log")
    print "Stochastic Gradient Classifier"
    prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_nam)
    '''

# Prediction algorithm to apply co-training algorithm using specified classifier
def prediction_func(clf1, clf2, clf3, feature1, feature2, sizes, train, test, label_name1):

    # For different proportions of labeled and unlabeled training data
    for size_test in sizes:
        print "Unlabeled Training Data Size: ", size_test
        train_labeled, train_unlabeled = semi_supervised_split(train, size_test)
        count = 0
        accuracy_list = []
        precision_list = []
        recall_list = []
        fscore_list = []
        data1 = pd.DataFrame(train_labeled[feature1])
        data2 = pd.DataFrame(train_labeled[feature2])
        test1 = pd.DataFrame(train_unlabeled[feature1])
        test2 = pd.DataFrame(train_unlabeled[feature2])

        val1 = 0
        val2 = 1

        # Check if there is an increase in correct predictions OR the test (unlabeled training) dataset is empty
        while val1 != val2 and len(test1)!=0:

            test_check1 = pd.DataFrame(columns=feature1)
            test_check2 = pd.DataFrame(columns=feature2)

            val2 = val1

            max1 = []
            max2 = []

            test1_data = test1.iloc[:, :-1]
            test2_data = test2.iloc[:, :-1]

            test1_labels = test1.iloc[:, -1]
            test2_labels = test2.iloc[:, -1]

            # Train with labeled data and predict using unlabeled data using classifier 1
            clf1.fit(data1.ix[:, :-1], data1.ix[:, -1])
            prediction1 = clf1.predict(test1_data)
            prediction1 = pd.DataFrame(prediction1)
            prediction1.columns = [label_name1]

            # Train with labeled data and predict using unlabeled data using classifier 1
            clf2.fit(data2.ix[:, :-1], data2.ix[:, -1])
            prediction2 = clf2.predict(test2_data)
            prediction2 = pd.DataFrame(prediction2)
            prediction2.columns = [label_name1]

            # Obtain probability of classification
            probability1 = clf1.predict_log_proba(test1_data)
            probability2 = clf2.predict_log_proba(test2_data)

            for i in range(len(probability1)):
                max1.append(max(probability1[i]))
                max2.append(max(probability2[i]))

            max1 = pd.DataFrame(max1)
            max1.columns = ["prob"]
            max2 = pd.DataFrame(max2)
            max2.columns = ["prob"]

            test1_data.reset_index(drop=True, inplace=True)
            max1.reset_index(drop=True, inplace=True)
            prediction1.reset_index(drop=True, inplace=True)
            temp1 = pd.concat([test1_data, prediction1, max1], axis=1)
            temp1.sort_values("prob", ascending=False)

            test2_data.reset_index(drop=True, inplace=True)
            max2.reset_index(drop=True, inplace=True)
            prediction2.reset_index(drop=True, inplace=True)
            temp2 = pd.concat([test2_data, prediction2, max2], axis=1)
            temp2.sort_values("prob", ascending=False)

            k = 0

            # For the top 5% and bottom 5% most confident predictions, add correct classifications to training labeled dataset
            for i in range(int(math.ceil(0.05*len(temp1)))):
                k = i
                if temp1.ix[i, label_name1] == temp2.ix[i, label_name1]:
                    temp1.reset_index(drop=True, inplace=True)
                    temp2.reset_index(drop=True, inplace=True)
                    data1 = data1.append(temp1.ix[i, :-1])
                    data2 = data2.append(temp2.ix[i, :-1])
                if temp1[label_name1].iloc[-i] == temp2[label_name1].iloc[-i]:
                    temp1.reset_index(drop=True, inplace=True)
                    temp2.reset_index(drop=True, inplace=True)
                    data1 = data1.append(temp1.iloc[-i, :-1])
                    data2 = data2.append(temp2.iloc[-i, :-1])
                else:
                    test_check1.reset_index(drop=True, inplace=True)
                    test_check2.reset_index(drop=True, inplace=True)
                    test_check1 = test_check1.append(temp1.ix[i, :-1])
                    test_check2 = test_check2.append(temp2.ix[i, :-1])

            for i in range(k+1, len(temp1)-k):
                test_check1.reset_index(drop=True, inplace=True)
                test_check2.reset_index(drop=True, inplace=True)
                test_check1 = test_check1.append(temp1.ix[i, :-1])
                test_check2 = test_check2.append(temp2.ix[i, :-1])

            # temp1 = temp1.iloc[:, :-2]
            # temp1.reset_index(drop=True, inplace=True)
            # temp2 = temp2.iloc[:, :-1]
            # temp2.reset_index(drop=True, inplace=True)
            # train_unlabeled = pd.concat([temp1, temp2], axis=1)

            test1 = test_check1
            test2 = test_check2

            data1.reset_index(drop=True, inplace=True)
            data2.reset_index(drop=True, inplace=True)

            train_labeled = pd.concat([data1.ix[:, :-1], data2], axis=1)

            val1 = len(train_labeled)

            train_labeled.reset_index(drop=True, inplace=True)
            test.reset_index(drop=True, inplace=True)

            # Train a classifier and predict using test data to check accuracy of current model
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
    feature_set1 = ["fixed acidity", "volatile acidity", "residual sugar", "total sulfur dioxide", "sulphates", "alcohol", "chlorides", "quality"]
    feature_set2 = ["citric acid", "free sulfur dioxide", "density", "pH", "quality"]
    train_data, test_data = train_test_split(wine_data, test_size=0.3)
    label_name = "quality"
    print
    print "WINE DATASET"
    function_loop(feature_set1, feature_set2, test_size, train_data, test_data, label_name)

    n = len(set(sensor_data.iloc[:, -1]))
    feature_set1 = ["f2", "f4", "f6", "f7", "f10", "f14", "f18", "f22", "f24", "class"]
    feature_set2 = ["f1", "f3", "f5", "f8", "f9", "f11", "f12", "f13", "f15", "f16", "f17", "f19", "f20", "f21", "f23", "class"]
    train_data, test_data = train_test_split(sensor_data, test_size=0.3)
    label_name = "class"
    print
    print "SENSOR DATASET"
    function_loop(feature_set1, feature_set2, test_size, train_data, test_data, label_name)

    n = len(set(waveform_data.iloc[:, -1]))
    feature_set1 = ["f1", "f2", "f3", "f4", "f6", "f10", "f12", "f13", "f14", "f15", "f18", "f20", "f21", "f22", "f24", "f26", "f28", "f32", "f35", "f36", "f37", "f40", "class"]
    feature_set2 = ["f5", "f7", "f8", "f9", "f11", "f16", "f17", "f19", "f23", "f25", "f27", "f29", "f30", "f31", "f33", "f34", "f38", "f39", "class"]
    train_data, test_data = train_test_split(waveform_data, test_size=0.3)
    label_name = "class"
    print
    print "WAVEFORM DATASET"
    function_loop(feature_set1, feature_set2, test_size, train_data, test_data, label_name)
