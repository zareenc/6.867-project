import argparse
from preprocess import Preprocessor
from six.moves import cPickle as pickle
import pdb

def save_pickle_file(pickle_file, save_dict):
    try:
        f = open(pickle_file, 'wb')
        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    print("Datasets saved to file", pickle_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Create serialized data sets.',
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
    parser.add_argument(
            'pickle_file_path',
            type=str,
            help='Path to test data.',
            )

    args = parser.parse_args()

    print("Loading training data")
    training_preprocessor = Preprocessor(args.training_data, verbose=True)
    training_preprocessor.cleanup()
    training_dictionary = training_preprocessor.get_dictionary()
    X_train, Y_train_multi, Y_train_binary = training_preprocessor.featurize(training_dictionary)

    del training_preprocessor

    print("Loading validation data")
    validation_preprocessor = Preprocessor(args.validation_data, verbose=True)
    validation_preprocessor.cleanup()
    X_val, Y_val_multi, Y_val_binary = validation_preprocessor.featurize(training_dictionary)

    del validation_preprocessor

    print("Loading testing data")
    testing_preprocessor = Preprocessor(args.test_data, verbose=True)
    testing_preprocessor.cleanup()
    X_test, Y_test_multi, Y_test_binary = testing_preprocessor.featurize(training_dictionary)

    del testing_preprocessor
    del training_dictionary

    pdb.set_trace()

    save = {
        'X_train': X_train,
        'Y_train_multi': Y_train_multi,
        'Y_train_binary': Y_train_binary,
        'X_val': X_val,
        'Y_val_multi': Y_val_multi,
        'Y_val_binary': Y_val_binary,
        'X_test': X_test,
        'Y_test_multi': Y_test_multi,
        'Y_test_binary': Y_test_binary
    }

    save_pickle_file(args.pickle_file_path, save)