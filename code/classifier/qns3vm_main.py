import numpy as np
import random
import copy
from qns3vm import QN_S3VM

#driver function for Quasi-Newton Semi-Supervised Support Vector Machines
def qns3vm_main(features_train, features_test, labels_train, labels_test, labelled_data_percent):

    features_train_labeled = []
    features_train_unlabelled = []
    labels_train_labeled = []
    #generating labelled and unlabelled data from train set
    for class_label in np.unique(labels_train):
        class_selected_index = np.where(labels_train==class_label)[0]
        selected_labels = random.sample(class_selected_index,int(max(len(class_selected_index)*labelled_data_percent, 2)))
        features_train_labeled.extend(features_train[selected_labels])
        features_train_unlabelled.extend(features_train[list(set(class_selected_index)-set(selected_labels))])
        labels_train_labeled.extend(labels_train[selected_labels])
    labels_train_labeled=np.array(labels_train_labeled)
    accuracy = 0
    precision = 0
    recall = 0
    #looping over unique class labels in test data since the implementation used can only handle two classes at a time we converted it to a one vs rest classifier
    for class_label in np.unique(labels_test):
        #a random generator since one is required by the implemetation being used
        my_random_generator = random.Random()
        my_random_generator.seed(0)
        #converting the labels of lablled data into 2 classes only
        labels_train_labeled_copy = copy.deepcopy(labels_train_labeled)
        labels_train_labeled[:] = -1
        labels_train_labeled[labels_train_labeled_copy==class_label]=1
        #creating a model of Quasi-Newton Semi-Supervised Support Vector Machines
        model = QN_S3VM(features_train_labeled, labels_train_labeled, features_train_unlabelled, my_random_generator, lam=0.0009765625, lamU=1, kernel_type="RBF", sigma=0.5,  estimate_r=0.0,)
        #training the model
        model.train()
        #generating the predictinos from the model on the test features
        preds = np.array(model.getPredictions(features_test))
        #calculating accuracies, precision and recall
        accuracy += sum(preds[labels_test==class_label]==1)/float(len(labels_test))
        labels_train_labeled = labels_train_labeled_copy
        precision += (sum(preds[labels_test==class_label]==1)/float(len(preds[labels_test==class_label])))
        if sum(preds==1)!=0:
            recall += (sum(preds[labels_test==class_label]==1)/float(sum(preds==1)))

    #normalizing precision and recall
    precision /= len(np.unique(labels_train))
    recall /= len(np.unique(labels_train))

    print 'Accuracy:', accuracy
    #precision and recall might be 0
    if precision+recall!=0:
        print 'F1 score:', (2*precision*recall)/(precision+recall)
    else:
        print 'F1 score:', 0
    print 'Precision:', precision
    print 'Recall:', recall


if __name__ == "__main__":
    main()