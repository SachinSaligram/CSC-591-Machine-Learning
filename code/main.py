import os
import csv
import numpy as np
import random
from sklearn.model_selection import train_test_split
from classifier.ann import AnnSingleLayer
from classifier.svm import SVM
from classifier.logistic_regression import LogisticRegressionClassifier
from classifier.qns3vm_main import qns3vm_main

#function to load selected dataset
def load_dataset(dataset_number):
    features = []
    labels = []
    if dataset_number == 1:
        features, labels = load_wine_dataset()
    elif dataset_number == 2:
        features, labels = load_waveform_dataset()
        unique_labels = np.unique(np.array(labels))
        #since the labels are strings converting them to float values
        labels = np.array([np.where(unique_labels==x)[0][0] for x in labels])
    else:
        features, labels = load_wallrobot_dataset()
        unique_labels = np.unique(np.array(labels))
        #since the labels are strings converting them to float values
        labels = np.array([np.where(unique_labels==x)[0][0] for x in labels])
    
    #changing the datatype of labels from float to int
    labels = labels.astype(np.int)
    #returns features and labels
    return features, labels

#function to load wine dataset
def load_wine_dataset():
    csvfile = open(os.path.join('dataset/winequality-white.csv'))
    winereader = csv.reader(csvfile, delimiter=';')
    return load_csv_dataset(winereader)

#function to load wave form dataset
def load_waveform_dataset():
    csvfile = open(os.path.join('dataset/waveform-+noise.data.csv'))
    waveformreader = csv.reader(csvfile, delimiter=';')
    return load_csv_dataset(waveformreader)

#function to load wall following robot dataset
def load_wallrobot_dataset():
    csvfile = open(os.path.join('dataset/sensor_readings_24.data.csv'))
    wallrobotreader = csv.reader(csvfile, delimiter=';')
    return load_csv_dataset(wallrobotreader)

#function to read the csv files and to separate features and labels into different numpy arrays
def load_csv_dataset(filereader):
    features = []
    labels = []
    num_of_features = 0
    #the first row gives us the number of features
    for row in filereader:
        num_of_features = len(row)
        break
    #rest of the lines of file gives us the featuers and their labels
    for row in filereader:
        temp = []
        for index in range(num_of_features-1):
            temp.append(float(row[index]))
        labels.append(row[num_of_features-1])
        features.append(np.array(temp))

    #converting features and labels list to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    #was for testing purpose
    '''print features
    print labels
    print {'features': features, 'labels':labels}
    return {'features': np.array(features), 'labels':np.array(labels)}'''

    #returning the extracted features and labels
    return features, labels


def main():
    while True:
        print "Select dataset by entering the number: "
        print "1. White wine"
        print "2. WaveForm"
        print "3. Wall-Following robot navigation"
        dataset_number = raw_input()
        try:
            dataset_number=int(dataset_number)
            if dataset_number>0 and dataset_number<4:
                break
            print "number out of range"
        except ValueError:
            print "not a valid number"

    features, labels = load_dataset(dataset_number) #dataset is a dictionary with features and labels as keys

    #splitting the data into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=.70, random_state=42)

    #setting a seed to get similar output if there is randomization anywhere ahead in the code
    random.seed(42)

    #looping over different ratio of labelled data
    for labelled_data_percent in [.10, .20, .30, .40, .50, .60, .70, .80, .90]:
        print 'For ' + str(labelled_data_percent*100) + '% of labelled data'
        #splitting into labeled and unlabelled sets
        features_train_labeled, features_train_unlabelled, labels_train_labeled, labels_train_unlabelled = train_test_split(features_train, labels_train, train_size=labelled_data_percent, random_state=42)

        #Supervised Algorithms
        
        #Artificial Neural Network
        #for ann we will use features_train_labeled, labels_train_labeled for traning and features_test, labels_test for testing
        print 'ANN'
        AnnSingleLayer(features_train_labeled, features_test, labels_train_labeled, labels_test)
        print '\n\n'

        #Support Vector Machines
        print 'SVM'
        SVM(features_train_labeled, features_test, labels_train_labeled, labels_test)
        print '\n\n'

        #Logistic Regression
        print 'Logistic Regression'
        LogisticRegressionClassifier(features_train_labeled, features_test, labels_train_labeled, labels_test)
        print '\n\n'


        #Semi supervised algorithms
        
        #Semi Supervised Support Vector Machine
        #it will use the main train and test features and labels data to be sent to function and split since we need to ensure that there is some train data for each class in the test data. It is the requirement of the implementation used.
        print 'S3VM'
        qns3vm_main(features_train, features_test, labels_train, labels_test,labelled_data_percent)
        print '\n\n'


if __name__ == "__main__":
    main()