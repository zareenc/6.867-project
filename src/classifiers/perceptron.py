import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../preprocessing')
import numpy as np
from sklearn.linear_model import perceptron
from preprocess import *
from helpers import *
import pdb


def train_perceptron(X_train, Y_train):
	p = perceptron.Perceptron()
	p.fit(X_train, np.ravel(Y_train))
	return p


def get_perceptron_error(linear_model, X, Y):
	n,d = X.shape
	return classif_err(expand(linear_model.predict(X)), Y)


if __name__ == "__main__":

	'''
	csv files to run on: (nv and az)
	
	train: filtered_nv_reviews_train.csv
	validation: filtered_nv_reviews_val.csv
	test: filtered_nv_reviews_test.csv
	
	train: filtered_az_reviews_train.csv
	validation: filtered_az_reviews_val.csv
	test: filtered_az_reviews_test.csv
	'''

	# get arguments
	parser = ClassificationParser()
	args = parser.parse_args()

	train_csv = args.train_file
	val_csv = args.val_file
	test_csv = args.test_file

	pre_train = Preprocessor(train_csv)
	pre_val = Preprocessor(val_csv)
	pre_test = Preprocessor(test_csv)

	pre_train.cleanup()
	dict_train = pre_train.get_dictionary()
	X_TRAIN, Y_TRAIN_MULTI, Y_TRAIN_BINARY = pre_train.featurize(dict_train)

	pre_val.cleanup()
	X_VAL, Y_VAL_MULTI, Y_VAL_BINARY = pre_val.featurize(dict_train)

	pre_test.cleanup()
	X_TEST, Y_TEST_MULTI, Y_TEST_BINARY = pre_test.featurize(dict_train)
	
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

