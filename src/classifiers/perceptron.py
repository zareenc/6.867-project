import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../preprocessing')
import numpy as np
from sklearn.linear_model import perceptron
from sklearn.model_selection import cross_val_predict
from preprocess import *
from helpers import *
import pdb


def train_perceptron(X_train, Y_train):
	p = perceptron.Perceptron(penalty='l2')
	p.fit(X_train, np.ravel(Y_train))
	return p

def get_perceptron_error(linear_model, X, Y):
	return classif_err(expand(linear_model.predict(X)), Y)

# def get_cross_validation_error(linear_model, X, Y, k):
# 	predictions = cross_val_predict(linear_model, X, np.ravel(Y), cv=k)
# 	return classif_err(expand(predictions), Y)


if __name__ == "__main__":

	# get arguments
	parser = ClassificationParser()
	args = parser.parse_args()
	multi_class = False				# Perceptron should always be binary

	train_csv = args.train_file
	test_csv = args.test_file

	features = ['city']

	pre_train = Preprocessor(train_csv, args.business_csv)
	pre_test = Preprocessor(test_csv, args.business_csv)

	pre_train.cleanup(modify_words_dictionary=True)
	dict_train = pre_train.get_words_dictionary()
	X_TRAIN, Y_TRAIN_BINARY = pre_train.featurize(dict_train, multi_class, feature_attributes_to_use=features)

	pre_test.cleanup()
	X_TEST, Y_TEST_BINARY = pre_test.featurize(dict_train, multi_class, feature_attributes_to_use=features)

	k = 20

	# training
	model = perceptron.Perceptron(penalty='l2')
	_, acc_val = get_cross_validation_error(model, X_TRAIN, Y_TRAIN_BINARY, k)
	print("Cross validation accuracy: ", acc_val)

	# test
	model = train_perceptron(X_TRAIN, Y_TRAIN_BINARY)
	_, acc_val = get_perceptron_error(model, X_TEST, Y_TEST_BINARY)
	print("test accuracy: ", acc_val)
