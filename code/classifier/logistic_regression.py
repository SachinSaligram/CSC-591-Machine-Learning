from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score

def LogisticRegressionClassifier(features_train, features_tets, labels_train, labels_test):
    
    #testing purpose
    '''#generating a model
    clf = LogisticRegression(multi_class='ovr')
    #fitting the model
    clf.fit(features_train, labels_train)

    dec = clf.decision_function(features_tets)
    print dec.shape[1] # 4 classes: 4*3/2 = 6

    clf.decision_function_shape = "ovr"
    dec = clf.decision_function(features_tets)
    print dec.shape[1] # 4 classes'''

    #A logistic classifier with random state, gave poor accuracies
    '''pred_y = OneVsRestClassifier(LogisticRegression(random_state=0)).fit(features_train, labels_train).predict(features_tets)
    print "Accuracy: %f" % (sum([1 for i in range(len(pred_y)) if pred_y[i]==labels_test[i]])/float(len(pred_y)))
    print "F1 score: %f" % f1_score(labels_test, pred_y, average='macro')
    print "Precision score: %f" % precision_score(labels_test, pred_y, average='macro')
    print "Recall score: %f" % recall_score(labels_test, pred_y, average='macro')'''

    #A logistic regression classifier with one vs rest feature used to fit in One VS Rest Classifier and used it to predict
    pred_y = OneVsRestClassifier(LogisticRegression(multi_class='ovr')).fit(features_train, labels_train).predict(features_tets)
    print "Accuracy: %f" % (sum([1 for i in range(len(pred_y)) if pred_y[i]==labels_test[i]])/float(len(pred_y)))
    print "F1 score: %f" % f1_score(labels_test, pred_y, average='macro')
    print "Precision score: %f" % precision_score(labels_test, pred_y, average='macro')
    print "Recall score: %f" % recall_score(labels_test, pred_y, average='macro')