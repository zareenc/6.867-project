import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../preprocessing')
import numpy as np
import argparse
import pdb
from sklearn.linear_model import LogisticRegression
from helpers import *
from get_yelp_data import get_review_data
from preprocess import *


'''
Logistic regression

Arguments
   X_*: ndarray with dimensions (n, d)
   Y_*: ndarray with dimensions (n, 1)
   k: The number of folds for cross validation
   params: tuple of (lambda, scaling factor, regularization, multi_class)

Return:
   tuple of (cross validation accuracy, test accuracy)
'''
def logistic_regression(X_train, Y_train, X_test, Y_test, k, *params):
    # initialization
    l, scaling, reg, multi_class = params
    C = 1./l
    lr = LogisticRegression(solver='liblinear', penalty=reg, \
             fit_intercept=True, intercept_scaling=scaling, C=C)
    if multi_class:
        params = {'solver':'lbfgs', 'multi_class':'multinomial'}
    else:
        params = {'solver':'liblinear'}
    lr.set_params(**params)
    
    # training
    _, acc_cv = get_cross_validation_error(lr, X_train, Y_train, k)
    print("cross validation accuracy: %f" % acc_cv)

    # testing
    lr.fit(X_train, np.ravel(Y_train))
    pred_test = lr.predict(X_test)
    
    _, acc_test = classif_err(expand(pred_test), Y_test)
    print("Test accuracy: %f\n" % acc_test)

    return (acc_cv, acc_test)

if __name__ == "__main__":

    # parser of command line args
    parser = ClassificationParser()

    # get arguments
    args = parser.parse_args()
    train_csv_file = args.train_file
    test_csv_file = args.test_file
    multi_class = args.multi_class
    frequency = args.frequency
    tf_idf = args.tf_idf

    features = ['city']

    # clean up reviews
    preprocessor_train = Preprocessor(train_csv_file, args.business_csv)
    preprocessor_test = Preprocessor(test_csv_file, args.business_csv)
    print("cleaning up reviews...")
    preprocessor_train.cleanup(modify_words_dictionary=True)
    preprocessor_test.cleanup()

    # featurize training and test data
    print("featurizing training and test data...")
    train_dict = preprocessor_train.get_words_dictionary()
    X_train, Y_train = preprocessor_train.featurize(train_dict, multi_class, frequency=frequency, tf_idf=tf_idf, feature_attributes_to_use=features)
    X_test, Y_test = preprocessor_test.featurize(train_dict, multi_class, frequency=frequency, tf_idf=tf_idf, feature_attributes_to_use=features)

    # run logistic regression
    lambdas = [0.01, 0.1, 0.5, 1., 5., 10.]
    scaling = 100.
    k = 5
    if multi_class:
        regs = ['l2']
    else:
        regs = ['l1', 'l2']
    params_list = []
    results_list = []
    params_best = {}
    results_best = {}
    for reg in regs:
        for l in lambdas:
            params_dict = {'lambda':l, 'scaling factor':scaling, 'regularization':reg, 'multi_class':multi_class}
            params = (l, scaling, reg, multi_class)
            acc_cv, acc_test = logistic_regression( \
                                       X_train, Y_train, \
                                       X_test, Y_test, k, *params)
            results_dict = {'cv acc':acc_cv, 'test acc':acc_test}

            if len(results_best) == 0 or acc_cv > results_best['cv acc']:
                params_best = params_dict
                results_best = results_dict
            params_list.append(params_dict)
            results_list.append(results_dict)
    
    print("best parameters", params_best)
    print("best results", results_best)
