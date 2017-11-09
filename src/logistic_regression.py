import numpy as np
import argparse
import pdb
from sklearn.linear_model import LogisticRegression
from get_yelp_data import get_review_data
from preprocess import *
from helpers import *


'''
Logistic regression

Arguments
   X_*: ndarray with dimensions (n, d)
   Y_*: ndarray with dimensions (n, 1)
   params: tuple of (lambda, scaling factor, regularization)

Return:
   tuple of (training accuracy, validation accuracy, test accuracy)
'''
def logistic_regression(X_train, Y_train, X_val, Y_val, \
                            X_test, Y_test, *params):
    # initialization
    l, scaling, reg = params
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
    err_train, acc_train = binary_classif_err(pred_train.reshape((n_train, 1)), Y_train)
    print "Training accuracy: %f" % acc_train
    
    # validation 
    print "validation. . ."
    n_val, d = X_val.shape
    pred_val = lr.predict(X_val)
    err_val, acc_val = binary_classif_err(pred_val.reshape((n_val, 1)), Y_val)
    print "Validation accuracy: %f" % acc_val

    # testing
    print "testing. . ."
    n_test, d = X_test.shape
    pred_test = lr.predict(X_test)
    err_test, acc_test = binary_classif_err(pred_test.reshape((n_test, 1)), Y_test)
    print "Test accuracy: %f\n" % acc_test

    return (acc_train, acc_val, acc_test)

if __name__ == "__main__":
    train_csv_file = 'data/filtered_az_reviews_train.csv'
    val_csv_file = 'data/filtered_az_reviews_val.csv'
    test_csv_file = 'data/filtered_az_reviews_test.csv'
    results_file = 'results/filtered_az_reviews_lr.txt'

    # clean up reviews
    print "cleaning up reviews"
    preprocessor_train = Preprocessor(train_csv_file)
    preprocessor_val = Preprocessor(val_csv_file)
    preprocessor_test = Preprocessor(test_csv_file)
    preprocessor_train.cleanup()
    preprocessor_val.cleanup()
    preprocessor_test.cleanup()

    # featurize training, validation, test data
    print "featurizing training, validation, test data"
    train_dict = preprocessor_train.get_dictionary()
    X_train, Y_train_multi, Y_train_binary = preprocessor_train.featurize(train_dict)
    X_val, Y_val_multi, Y_val_binary = preprocessor_val.featurize(train_dict)
    X_test, Y_test_multi, Y_test_binary = preprocessor_test.featurize(train_dict)

    # run logistic regression
    lambdas = [0.01, 0.1, 0.5, 1., 5., 10.]
    scaling = 100.
    regs = ['l1', 'l2']
    params_list = []
    results_list = []
    params_best = {}
    results_best = {}
    for reg in regs:
        for l in lambdas:
            params_dict = {'lambda':l, 'scaling factor':scaling, 'regularization':reg}
            params = (l, scaling, reg)
            acc_train, acc_val, acc_test = logistic_regression( \
                                       X_train, Y_train_binary, \
                                       X_val, Y_val_binary, \
                                       X_test, Y_test_binary, *params)
            results_dict = {'train acc':acc_train, 'val acc':acc_val, 'test acc':acc_test}

            if len(results_best) == 0 or acc_val > results_best['val acc']:
                params_best = params_dict
                results_best = results_dict
            params_list.append(params_dict)
            results_list.append(results_dict)
    
    write_results(results_file, params_list, results_list)
    print "best parameters", params_best
    print "best results", results_best
