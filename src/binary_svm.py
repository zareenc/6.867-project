import numpy as np
import pdb
from sklearn import svm
from preprocess import Preprocessor
import argparse

def count_errors(set1, set2):
    errors = 0
    for i in range(len(set1)):
        if set1[i] != set2[i]:
            errors += 1
    return errors, float(len(set1) - errors)/len(set1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Run binary SVM.',
            )
    parser.add_argument(
            'training_data',
            type=str,
            help='Path to training data.',
            )
    parser.add_argument(
            '--validation_data',
            type=str,
            help='Path to validation data.',
            )
    parser.add_argument(
            'test_data',
            type=str,
            help='Path to test data.',
            )

    args = parser.parse_args()

    print "Loading training data"
    training_preprocessor = Preprocessor(args.training_data)
    training_preprocessor.cleanup()
    training_dictionary = training_preprocessor.get_dictionary()
    X_train, _, Y_train_binary = training_preprocessor.featurize(training_dictionary)

    if args.validation_data:
        print "Loading validation data"
        training_preprocessor = Preprocessor(args.validation_data)
        training_preprocessor.cleanup()
        X_val, _, Y_val_binary = training_preprocessor.featurize(training_dictionary)

    print "Loading testing data"
    testing_preprocessor = Preprocessor(args.test_data)
    testing_preprocessor.cleanup()
    X_test, _, Y_test_binary = testing_preprocessor.featurize(training_dictionary)

    print "Running SVM"
    max_accuracy = 0
    best_C = 0
    for C in [0.01, 0.1, 1, 5, 10]: 
        binary_classifier = svm.SVC(C=C)
        binary_classifier.fit(X_train, np.ravel(Y_train_binary))
        Y_predict = binary_classifier.predict(X_val)
        val_errs, val_accuracy = count_errors(Y_predict, Y_val_binary)
        print "C =", C, ", gamma= ", gamma, ", validation errors:", val_errs, ", validation accuracy:", val_accuracy
        if val_accuracy > max_accuracy:
            max_accuracy = val_accuracy
            best_C = C

    print "best C:", best_C

    best_classifier = svm.SVC(best_C)
    Y_predict = binary_classifier.predict(X_test)

    test_errs, test_accuracy = count_errors(Y_predict, Y_test_binary)
    print "total test errors:", test_errs
    print "total test accuracy:", test_accuracy


