import numpy as np
import argparse
import pdb
from get_yelp_data import get_review_data
from preprocess import *
from sklearn.linear_model import LogisticRegression
from helpers import *


'''Logistic regression'''
def logistic_regression_binary(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    # initialization
    reg = 'l2'
    scaling = 100.
    l = 0.001
    C = 1./l
    lr = LogisticRegression(solver='liblinear', penalty=reg, \
             fit_intercept=True, intercept_scaling=scaling, C=C)
    
    # training
    print "training. . ."
    n_train, d = X_train.shape
    lr.fit(X_train, Y_train.reshape(n_train,))
    w = lr.coef_
    w_o = lr.intercept_
    print "weight vector:", w
    print "bias:", w_o
    pred_train = lr.predict(X_train)
    err_train = binary_classif_err(pred_train.reshape((n_train, 1)), Y_train)
    acc_train = (n_train - err_train) / n_train
    print "Training accuracy: %f" % acc_train
    
    # validation 
    print "validation. . ."
    n_val, d = X_val.shape
    pred_val = lr.predict(X_val)
    err_val = binary_classif_err(pred_val.reshape((n_val, 1)), Y_val)
    acc_val = (n_val - err_val) / n_val
    print "Validation accuracy: %f" % acc_val

    # testing
    print "testing. . ."
    n_test, d = X_test.shape
    pred_test = lr.predict(X_test)
    err_test = binary_classif_err(pred_test.reshape((n_test, 1)), Y_test)
    acc_test = (n_test - err_test) / n_test
    print "Test accuracy: %f" % acc_test

if __name__ == "__main__":
    train_csv_file = 'data/filtered_az_reviews_train.csv'
    val_csv_file = 'data/filtered_az_reviews_val.csv'
    test_csv_file = 'data/filtered_az_reviews_test.csv'

    # clean up reviews
    preprocessor_train = Preprocessor(train_csv_file)
    preprocessor_train.cleanup()
    preprocessor_val = Preprocessor(val_csv_file)
    preprocessor_val.cleanup()
    preprocessor_test = Preprocessor(test_csv_file)
    preprocessor_test.cleanup()

    # get training, validation, test data
    print "getting training, validation, test data"
    dict = preprocessor_train.get_dictionary()
    X_train, Y_train_multi, Y_train_binary = preprocessor_train.featurize(dict)
    X_val, Y_val_multi, Y_val_binary = preprocessor_val.featurize(dict)
    X_test, Y_test_multi, Y_test_binary = preprocessor_test.featurize(dict)

    # run logistic regression
    logistic_regression_binary(X_train, Y_train_binary, \
                               X_val, Y_val_binary, \
                               X_test, Y_test_binary)
