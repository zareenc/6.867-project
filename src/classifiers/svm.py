import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../preprocessing')
import numpy as np
import pdb
import argparse
from sklearn import svm
from six.moves import cPickle as pickle
from preprocess import Preprocessor
from helpers import *

def binary_linear_svm_cv(Cees, X_train, Y_train_binary, X_test, Y_test_binary, k):
    print("")
    print("Running Binary Linear SVM")
    max_accuracy = 0
    best_C = 0
    best_classifier = None
    for C in Cees:
            binary_classifier = svm.SVC(C=C, kernel='linear')
            _, acc_cv = get_cross_validation_error(binary_classifier, X_train, Y_train_binary, k)
            print("C =", C, ", cross validation accuracy:", acc_cv)
            if acc_cv > max_accuracy:
                max_accuracy = acc_cv
                best_C = C
                best_classifier = binary_classifier

    print("best C:", best_C, ", best cross validation accuracy:", max_accuracy)

    best_classifier.fit(X_train, np.ravel(Y_train_binary))
    Y_predict = best_classifier.predict(X_test)

    _, test_accuracy = classif_err(expand(Y_predict), expand(Y_test_binary))
    print("total test accuracy:", test_accuracy)
    print("")


def binary_rbf_svm_cv(Cees, gammas, X_train, Y_train_binary, X_test, Y_test_binary, k):
    print("")
    print("Running Binary RBF SVM")
    max_accuracy = 0
    best_C = 0
    best_classifier = None
    for C in Cees:
        for gamma in gammas:
            binary_classifier = svm.SVC(C=C, gamma=gamma)
            _, acc_cv = get_cross_validation_error(binary_classifier, X_train, Y_train_binary, k)
            print("C =", C, ", gamma =", gamma, ", cross validation accuracy:", acc_cv)
            if acc_cv > max_accuracy:
                max_accuracy = acc_cv
                best_C = C
                best_gamma = gamma
                best_classifier = binary_classifier

    print("best C:", best_C, ", best gamma:", best_gamma, ", best cross validation accuracy:", max_accuracy)

    best_classifier.fit(X_train, np.ravel(Y_train_binary))
    Y_predict = best_classifier.predict(X_test)

    _, test_accuracy = classif_err(expand(Y_predict), expand(Y_test_binary))
    print("total test accuracy:", test_accuracy)
    print("")

def multiclass_linear_svm_cv(Cees, X_train, Y_train_multi, X_test, Y_test_multi, k):
    print("")
    print("Running Multiclass Linear SVM")
    max_accuracy = 0
    best_C = 0
    best_classifier = None
    for C in Cees:
            classifier = svm.SVC(C=C, kernel='linear', decision_function_shape='ovo')
            _, acc_cv = get_cross_validation_error(classifier, X_train, Y_train_multi, k)
            print("C =", C, ", cross validation accuracy:", acc_cv)
            if acc_cv > max_accuracy:
                max_accuracy = acc_cv
                best_C = C
                best_classifier = classifier

    print("best C:", best_C, ", best cross validation accuracy:", max_accuracy)

    best_classifier.fit(X_train, np.ravel(Y_train_multi))
    Y_predict = best_classifier.predict(X_test)

    _, test_accuracy = classif_err(expand(Y_predict), expand(Y_test_multi))
    print("total test accuracy:", test_accuracy)
    print("")


def multiclass_rbf_svm_cv(Cees, gammas, X_train, Y_train_multi, X_test, Y_test_multi, k):
    print("")
    print("Running Multiclass RBF SVM")
    max_accuracy = 0
    best_C = 0
    best_gamma = 0
    best_classifier = None
    for C in Cees:
        for gamma in gammas:
            classifier = svm.SVC(C=C, gamma=gamma, decision_function_shape='ovo')
            _, acc_cv = get_cross_validation_error(classifier, X_train, Y_train_multi, k)
            print("C =", C, ", gamma =", gamma, ", cross validation accuracy:", acc_cv)
            if acc_cv > max_accuracy:
                max_accuracy = acc_cv
                best_C = C
                best_gamma = gamma
                best_classifier = classifier

    print("best C:", best_C, ", best gamma:", best_gamma, ", best cross validation accuracy:", max_accuracy)

    best_classifier.fit(X_train, np.ravel(Y_train_multi))
    Y_predict = best_classifier.predict(X_test)

    _, test_accuracy = classif_err(expand(Y_predict), expand(Y_test_multi))
    print("test accuracy:", test_accuracy)
    print("")

if __name__ == "__main__":
    parser = ClassificationParser()

    parser.add_argument(
            '--norun_bin_lin',
            action='store_true',
            help='Whether to run binary linear SVM.',
            )
    parser.add_argument(
            '--norun_bin_rbf',
            action='store_true',
            help='Whether to run binary RBF SVM.',
            )
    parser.add_argument(
            '--norun_multi_lin',
            action='store_true',
            help='Whether to run multiclass linear SVM.',
            )
    parser.add_argument(
            '--norun_multi_rbf',
            action='store_true',
            help='Whether to run multiclass RBF SVM.',
            )

    args = parser.parse_args()
    multi_class = args.multi_class

    features = ['city']

    print("Loading training data")
    training_preprocessor = Preprocessor(args.train_file, args.business_csv)
    training_preprocessor.cleanup(modify_words_dictionary=True)
    training_dictionary = training_preprocessor.get_words_dictionary()
    X_train, Y_train = training_preprocessor.featurize(training_dictionary, multi_class, feature_attributes_to_use=features)
    
    print("Loading testing data")
    testing_preprocessor = Preprocessor(args.test_file, args.business_csv)
    testing_preprocessor.cleanup()
    X_test, Y_test = testing_preprocessor.featurize(training_dictionary, multi_class, feature_attributes_to_use=features)

    Cees = [0.01, 0.1, 1, 5]
    gammas = []
    for exp in range(-5, -3):
            gammas.append(2**exp)

    k = 5

    if multi_class:
        if not args.norun_multi_lin:
            multiclass_linear_svm_cv(Cees, X_train, Y_train, X_test, Y_test, k)
        if not args.norun_multi_rbf:
            multiclass_rbf_svm_cv(Cees, gammas, X_train, Y_train, X_test, Y_test, k)

    else:
        if not args.norun_bin_lin:
            binary_linear_svm_cv(Cees, X_train, Y_train, X_test, Y_test, k)
        if not args.norun_bin_rbf:
            binary_rbf_svm_cv(Cees, gammas, X_train, Y_train, X_test, Y_test, k)

