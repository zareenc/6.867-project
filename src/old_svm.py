import numpy as np
import pdb
import argparse
from sklearn import svm
from six.moves import cPickle as pickle
from preprocess import Preprocessor
from helpers import classif_err, expand, load_pickled_dataset

def binary_linear_svm(Cees, X_train, Y_train_binary, X_val, Y_val_binary, X_test, Y_test_binary):
    print
    print("Running Binary Linear SVM")
    max_accuracy = 0
    best_C = 0
    best_classifier = None
    for C in Cees:
            binary_classifier = svm.SVC(C=C, kernel='linear')
            binary_classifier.fit(X_train, np.ravel(Y_train_binary))
            Y_predict = binary_classifier.predict(X_val)
            val_errs, val_accuracy = classif_err(expand(Y_predict), expand(Y_val_binary))
            print("C =", C, ", validation errors:", val_errs, ", validation accuracy:", val_accuracy)
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                best_C = C
                best_classifier = binary_classifier

    print("best C:", best_C)

    Y_predict = best_classifier.predict(X_train)
    train_errs, train_accuracy = classif_err(expand(Y_predict), expand(Y_train_binary))
    print("total training errors:", train_errs)
    print("total training accuracy", train_accuracy)

    Y_predict = best_classifier.predict(X_test)

    test_errs, test_accuracy = classif_err(expand(Y_predict), expand(Y_test_binary))
    print("total test errors:", test_errs)
    print("total test accuracy:", test_accuracy)
    print


def binary_rbf_svm(Cees, gammas, X_train, Y_train_binary, X_val, Y_val_binary, X_test, Y_test_binary):
    print("")
    print("Running Binary RBF SVM")
    max_accuracy = 0
    best_C = 0
    best_classifier = None
    for C in Cees:
        for gamma in gammas:
            binary_classifier = svm.SVC(C=C, gamma=gamma)
            binary_classifier.fit(X_train, np.ravel(Y_train_binary))
            Y_predict = binary_classifier.predict(X_val)
            val_errs, val_accuracy = classif_err(expand(Y_predict), expand(Y_val_binary))
            print("C =", C, ", gamma =", gamma, ", validation errors:", val_errs, ", validation accuracy:", val_accuracy)
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                best_C = C
                best_gamma = gamma
                best_classifier = binary_classifier

    print("best C:", best_C, ", best gamma:", best_gamma)

    Y_predict = best_classifier.predict(X_train)
    train_errs, train_accuracy = classif_err(expand(Y_predict), expand(Y_train_binary))
    print("total training errors:", train_errs)
    print("total training accuracy", train_accuracy)

    Y_predict = best_classifier.predict(X_test)

    test_errs, test_accuracy = classif_err(expand(Y_predict), expand(Y_test_binary))
    print("total test errors:", test_errs)
    print("total test accuracy:", test_accuracy)
    print("")

def multiclass_linear_svm(Cees, X_train, Y_train_multi, X_val, Y_val_multi, X_test, Y_test_multi):
    print("")
    print("Running Multiclass Linear SVM")
    max_accuracy = 0
    best_C = 0
    best_classifier = None
    for C in Cees:
            classifier = svm.SVC(C=C, kernel='linear', decision_function_shape='ovo')
            classifier.fit(X_train, np.ravel(Y_train_multi))
            Y_predict = classifier.predict(X_val)
            val_errs, val_accuracy = classif_err(expand(Y_predict), expand(Y_val_multi))
            print("C =", C, ", validation errors:", val_errs, ", validation accuracy:", val_accuracy)
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                best_C = C
                best_classifier = classifier

    print("best C:", best_C)

    Y_predict = best_classifier.predict(X_train)
    train_errs, train_accuracy = classif_err(expand(Y_predict), expand(Y_train_multi))
    print("total training errors:", train_errs)
    print("total training accuracy", train_accuracy)

    Y_predict = best_classifier.predict(X_test)

    test_errs, test_accuracy = classif_err(expand(Y_predict), expand(Y_test_multi))
    print("total test errors:", test_errs)
    print("total test accuracy:", test_accuracy)
    print("")


def multiclass_rbf_svm(Cees, gammas, X_train, Y_train_multi, X_val, Y_val_multi, X_test, Y_test_multi):
    print("")
    print("Running Multiclass RBF SVM")
    max_accuracy = 0
    best_C = 0
    best_gamma = 0
    best_classifier = None
    for C in Cees:
        for gamma in gammas:
            classifier = svm.SVC(C=C, gamma=gamma, decision_function_shape='ovo')
            classifier.fit(X_train, np.ravel(Y_train_multi))
            Y_predict = classifier.predict(X_val)
            val_errs, val_accuracy = classif_err(expand(Y_predict), expand(Y_val_multi))
            print("C =", C, ", gamma =", gamma, ", validation errors:", val_errs, ", validation accuracy:", val_accuracy)
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                best_C = C
                best_gamma = gamma
                best_classifier = classifier

    print("best C:", best_C, ", best gamma:", best_gamma)

    Y_predict = best_classifier.predict(X_train)
    train_errs, train_accuracy = classif_err(expand(Y_predict), expand(Y_train_multi))
    print("total training errors:", train_errs)
    print("total training accuracy", train_accuracy)

    Y_predict = best_classifier.predict(X_test)

    test_errs, test_accuracy = classif_err(expand(Y_predict), expand(Y_test_multi))
    print("total test errors:", test_errs)
    print("total test accuracy:", test_accuracy)
    print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Run SVM.',
            )
    parser.add_argument(
            'training_data',
            type=str,
            help='Path to training data.',
            )
    parser.add_argument(
            'validation_data',
            type=str,
            help='Path to validation data.',
            )
    parser.add_argument(
            'test_data',
            type=str,
            help='Path to test data.',
            )
    # parser.add_argument(
    #         'pickle_file_path',
    #         type=str,
    #         help='Path to test data.',
    #         )
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

    print("Loading training data")
    training_preprocessor = Preprocessor(args.training_data)
    training_preprocessor.cleanup()
    training_dictionary = training_preprocessor.get_dictionary()
    X_train, Y_train_multi, Y_train_binary = training_preprocessor.featurize(training_dictionary)

    del training_preprocessor

    print("Loading validation data")
    validation_preprocessor = Preprocessor(args.validation_data)
    validation_preprocessor.cleanup()
    X_val, Y_val_multi, Y_val_binary = validation_preprocessor.featurize(training_dictionary)

    del validation_preprocessor

    print("Loading testing data")
    testing_preprocessor = Preprocessor(args.test_data)
    testing_preprocessor.cleanup()
    X_test, Y_test_multi, Y_test_binary = testing_preprocessor.featurize(training_dictionary)

    del testing_preprocessor
    del training_dictionary

    # pdb.set_trace()

    Cees = [0.01, 0.1, 1, 5, 10]
    gammas = []
    for exp in range(-5, 6):
            gammas.append(2**exp)

    if not args.norun_bin_lin:
        binary_linear_svm(Cees, X_train, Y_train_binary, X_val, Y_val_binary, X_test, Y_test_binary)
    if not args.norun_bin_rbf:
        binary_rbf_svm(Cees, gammas, X_train, Y_train_binary, X_val, Y_val_binary, X_test, Y_test_binary)
    if not args.norun_multi_lin:
        multiclass_linear_svm(Cees, X_train, Y_train_multi, X_val, Y_val_multi, X_test, Y_test_multi)
    if not args.norun_multi_rbf:
        multiclass_rbf_svm(Cees, gammas, X_train, Y_train_multi, X_val, Y_val_multi, X_test, Y_test_multi)

