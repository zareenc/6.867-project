import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import perceptron
#from sklearn.linear_model import SGDClassifier
from preprocess import *
import pdb

# testing on toy AZ and NV datasets - AZ has 132 reviews, NV has 60
training_csv = 'data/filtered_az_reviews.csv'
test_csv = 'data/filtered_nv_reviews.csv'

train_pre = Preprocessor(training_csv)
test_pre = Preprocessor(test_csv)

train_pre.cleanup()
train_dict = train_pre.get_dictionary()
X_TRAIN, Y_TRAIN_MULTI, Y_TRAIN_BINARY = train_pre.featurize(train_dict)

test_pre.cleanup()
X_TEST, Y_TEST_MULTI, Y_TEST_BINARY = test_pre.featurize(train_dict)



def train_perceptron(X_train, Y_train):
	p = perceptron.Perceptron()
	p.fit(X_train, Y_train)
	return p

def get_score(linear_model, X_test, Y_test):
	return linear_model.score(X_test, Y_test)


if __name__ == "__main__":
	
	model = train_perceptron(X_TRAIN, Y_TRAIN_BINARY)
	w = model.coef_
	w0 = model.intercept_
	print "w is ", w
	print "w0 is ", w0

	print get_score(model, X_TEST, Y_TEST_BINARY)