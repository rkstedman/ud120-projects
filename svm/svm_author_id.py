#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# These lines effectively slice the training dataset down to 1% of its
# original size, tossing out 99% of the training data.
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


C_trials = [10000.0] #[1.0, 10.0, 100.0, 1000.0, 10000.0]
for C in C_trials:
    print "fitting with C:", C
    # classifier = SVC(kernel='linear')
    # change kernel to rbf, more complex
    classifier = SVC(C=C,kernel='rbf',gamma='auto')

    t0 = time()
    classifier.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    t1 = time()
    predictions = classifier.predict(features_test)
    print "prediction time:", round(time()-t1, 3), "s"

    print "prediction for 10", predictions[10]
    print "prediction for 26", predictions[26]
    print "prediction for 50", predictions[50]

    print "total predictions", len(predictions)
    print predictions

    chrisPredictions = sum(predictions)
    sarahPredictions = len(predictions) - chrisPredictions
    print "Chris (1):", chrisPredictions
    print "Sarah (0):", sarahPredictions

    print "accuracy:", classifier.score(features_test, labels_test)


#########################################################
### Output ###
# with kernel = linear
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884
# training time: 173.906 s
# prediction time: 17.974 s
# accuracy: 0.9840728100113766

# When we decrease the training data by 99%
# training time: 0.098 s
# prediction time: 1.036 s
# accuracy: 0.8845278725824801

# when we change kernel to rbf
# training time: 0.111 s
# prediction time: 1.147 s
# accuracy: 0.6160409556313993

#########################################################
