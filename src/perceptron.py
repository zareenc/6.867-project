import numpy as np
from sklearn.linear_model import perceptron
#from sklearn.linear_model import SGDClassifier
from preprocess import *
from helpers import classif_err
import pdb


def train_perceptron(X_train, Y_train):
	p = perceptron.Perceptron()
	p.fit(X_train, Y_train)
	return p


def get_perceptron_error(linear_model, X, Y):
	n,d = X.shape
	return classif_err(linear_model.predict(X).reshape((n, 1)), Y)


if __name__ == "__main__":

	train_csv = '../data/filtered_nv_reviews_train.csv'
	val_csv = '../data/filtered_nv_reviews_val.csv'
	test_csv = '../data/filtered_nv_reviews_test.csv'

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
	print "w is ", w
	print "w0 is ", w0

	err_val, acc_val = get_perceptron_error(model, X_TRAIN, Y_TRAIN_BINARY)
	print "training errors: ", err_val
	print "training accuracy: ", acc_val

	# validation
	err_val, acc_val = get_perceptron_error(model, X_VAL, Y_VAL_BINARY)
	print "validation errors: ", err_val
	print "validation accuracy: ", acc_val

	# test
	err_val, acc_val = get_perceptron_error(model, X_TEST, Y_TEST_BINARY)
	print "test errors: ", err_val
	print "test accuracy: ", acc_val

