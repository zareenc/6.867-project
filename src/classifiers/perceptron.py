import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../preprocessing')
import numpy as np
from sklearn.linear_model import perceptron
from preprocess import *
from helpers import *
import pdb


def train_perceptron(X_train, Y_train):
	p = perceptron.Perceptron(penalty='l2')
	p.fit(X_train, np.ravel(Y_train))
	return p


def get_perceptron_error(linear_model, X, Y):
	n,d = X.shape
	return classif_err(expand(linear_model.predict(X)), Y)


if __name__ == "__main__":

	# get arguments
	parser = ClassificationParser()
	args = parser.parse_args()
	multi_class = False				# Perceptron should always be binary

	train_csv = args.train_file
	val_csv = args.val_file
	test_csv = args.test_file

	features = ['city']

	pre_train = Preprocessor(train_csv, args.business_csv)
	pre_val = Preprocessor(val_csv, args.business_csv)
	pre_test = Preprocessor(test_csv, args.business_csv)

	pre_train.cleanup(modify_words_dictionary=True)
	dict_train = pre_train.get_words_dictionary()
	X_TRAIN, Y_TRAIN_BINARY = pre_train.featurize(dict_train, multi_class, feature_attributes_to_use=features)

	pre_val.cleanup()
	X_VAL, Y_VAL_BINARY = pre_val.featurize(dict_train, multi_class, feature_attributes_to_use=features)

	pre_test.cleanup()
	X_TEST, Y_TEST_BINARY = pre_test.featurize(dict_train, multi_class, feature_attributes_to_use=features)
	
	# training
	model = train_perceptron(X_TRAIN, Y_TRAIN_BINARY)
	w = model.coef_
	w0 = model.intercept_
	print("w is ", w)
	print("w0 is ", w0)

	err_val, acc_val = get_perceptron_error(model, X_TRAIN, Y_TRAIN_BINARY)
	print("training errors: ", err_val)
	print("training accuracy: ", acc_val)

	# validation
	err_val, acc_val = get_perceptron_error(model, X_VAL, Y_VAL_BINARY)
	print("validation errors: ", err_val)
	print("validation accuracy: ", acc_val)

	# test
	err_val, acc_val = get_perceptron_error(model, X_TEST, Y_TEST_BINARY)
	print("test errors: ", err_val)
	print("test accuracy: ", acc_val)

