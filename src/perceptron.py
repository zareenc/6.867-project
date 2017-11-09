import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import perceptron
#from sklearn.linear_model import SGDClassifier
from preprocess import *
import pdb

# testing on toy AZ and NV datasets - AZ has 132 reviews, NV has 60
train_csv = 'data/filtered_az_reviews_train.csv'
val_csv = 'data/filtered_az_reviews_val.csv'
test_csv = 'data/filtered_az_reviews_test.csv'

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



def train_perceptron(X_train, Y_train):
	p = perceptron.Perceptron()
	p.fit(X_train, Y_train)
	return p

def get_perceptron_score(linear_model, X_test, Y_test):
	return linear_model.score(X_test, Y_test)


if __name__ == "__main__":
	
	model = train_perceptron(X_TRAIN, Y_TRAIN_BINARY)
	w = model.coef_
	w0 = model.intercept_
	print "w is ", w
	print "w0 is ", w0

	print "validation: ", get_perceptron_score(model, X_VAL, Y_VAL_BINARY)
	print "test: ", get_perceptron_score(model, X_TEST, Y_TEST_BINARY)