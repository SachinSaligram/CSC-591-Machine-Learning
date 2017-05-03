from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score, precision_score, f1_score

def SVM(features_train, features_tets, labels_train, labels_test):
    
    #was for testing purpose
    '''clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(features_train, labels_train)

    dec = clf.decision_function(features_tets)
    print dec.shape[1] # 4 classes: 4*3/2 = 6

    clf.decision_function_shape = "ovr"
    dec = clf.decision_function(features_tets)
    print dec.shape[1] # 4 classes'''


    #A linear SVM with random state, was giving similar results as the next one
    '''pred_y = OneVsRestClassifier(LinearSVC(random_state=0)).fit(features_train, labels_train).predict(features_tets)
    print "Accuracy: %f" % (sum([1 for i in range(len(pred_y)) if pred_y[i]==labels_test[i]])/float(len(pred_y)))
    print "F1 score: %f" % f1_score(labels_test, pred_y, average='macro')
    print "Precision score: %f" % precision_score(labels_test, pred_y, average='macro')
    print "Recall score: %f" % recall_score(labels_test, pred_y, average='macro')'''
    
    
    #A svm with one vs other passed to One vs Rest Classifier to fit on the data and predict results
    pred_y = OneVsRestClassifier(svm.SVC(decision_function_shape='ovo')).fit(features_train, labels_train).predict(features_tets)
    print "Accuracy: %f" % (sum([1 for i in range(len(pred_y)) if pred_y[i]==labels_test[i]])/float(len(pred_y)))
    print "F1 score: %f" % f1_score(labels_test, pred_y, average='macro')
    print "Precision score: %f" % precision_score(labels_test, pred_y, average='macro')
    print "Recall score: %f" % recall_score(labels_test, pred_y, average='macro')