import numpy as np
import pdb
import argparse
from sklearn import svm
from preprocess import Preprocessor
from helpers import binary_classif_err

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

    num_train_examples = X_train.shape[0]

    if args.validation_data:
        print "Loading validation data"
        training_preprocessor = Preprocessor(args.validation_data)
        training_preprocessor.cleanup()
        X_val, _, Y_val_binary = training_preprocessor.featurize(training_dictionary)

        num_val_examples = X_val.shape[0]

    print "Loading testing data"
    testing_preprocessor = Preprocessor(args.test_data)
    testing_preprocessor.cleanup()
    X_test, _, Y_test_binary = testing_preprocessor.featurize(training_dictionary)

    num_test_examples = X_test.shape[0]

    print "Running SVM"
    max_accuracy = 0
    best_C = 0
    for C in [0.01, 0.1, 1, 5, 10]: 
        binary_classifier = svm.SVC(C=C)
        binary_classifier.fit(X_train, np.ravel(Y_train_binary))
        Y_predict = binary_classifier.predict(X_val)
        val_errs, val_accuracy = binary_classif_err(Y_predict.reshape((num_val_examples, 1)), Y_val_binary.reshape((num_val_examples, 1)))
        print "C =", C, ", validation errors:", val_errs, ", validation accuracy:", val_accuracy
        if val_accuracy > max_accuracy:
            max_accuracy = val_accuracy
            best_C = C

    print "best C:", best_C

    best_classifier = svm.SVC(best_C)
    Y_predict = binary_classifier.predict(X_test)

    test_errs, test_accuracy = binary_classif_err(Y_predict.reshape((num_test_examples, 1)), Y_test_binary.reshape((num_test_examples, 1)))
    print "total test errors:", test_errs
    print "total test accuracy:", test_accuracy


