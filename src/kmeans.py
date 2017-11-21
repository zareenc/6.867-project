import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from preprocess import *
from helpers import classif_err
import pdb


def train_kmeans(X_train):
	kmeans = KMeans(n_clusters=5, init='k-means++')
	kmeans.fit(X_train)
	return kmeans

def get_comparison_tuples(predictions, labels_multi):
	compares = []
	for i in range(len(labels_multi)):
		compares.append((int(labels_multi[i]), predictions[i]))
	return compares

def dict_compare_tuples(compares):
	dict_compares = {}
	for i in range(len(compares)):
		if compares[i] not in dict_compares:
			dict_compares[compares[i]] = 0
		else:
			dict_compares[compares[i]] += 1
	#print "dict compares is: ", dict_compares
	return dict_compares

def eval_accuracy(dict_compares, classes):
	best_pairs = {}
	for i in range(1, classes+1):
		best_pairs[i] = 2*classes
	print "best pairs initialized to: ", best_pairs

	for i in range(1, classes+1):
		for tup in dict_compares:
			if tup[0] == i:
				if best_pairs[i] == 2*classes:
					best_pairs[i] = tup[1]
				elif dict_compares[(i, tup[1])] > dict_compares[(i, best_pairs[i])]:
					best_pairs[i] = tup[1]
	#print "FINAL BEST PAIRS: ", best_pairs
	
	best_pairs_set = set()
	for pred in best_pairs:
		best_pairs_set.add((pred, best_pairs[pred]))
	#print "BEST PAIRS SET: ", best_pairs_set

	err_num = 0.0
	err_den = 0.0

	for tup in dict_compares:
		err_den += dict_compares[tup]
		if tup not in best_pairs_set:
			err_num += dict_compares[tup]

	err_pct = err_num/(1.0*err_den)
	acc_pct = 1.0 - err_pct

	return err_pct, acc_pct


if __name__ == "__main__":

	train_csv = '../data/filtered_az_reviews_train.csv'
	val_csv = '../data/filtered_az_reviews_val.csv'
	test_csv = '../data/filtered_az_reviews_test.csv'

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
	model = train_kmeans(X_TRAIN)

	train_predictions = model.predict(X_TRAIN)
	train_comps = get_comparison_tuples(train_predictions, Y_TRAIN_MULTI)
	train_dict = dict_compare_tuples(train_comps)
	err, acc = eval_accuracy(train_dict, 5)
	print "Training error: ", err
	print "Training accuracy: ", acc

	val_predictions = model.predict(X_VAL)
	val_comps = get_comparison_tuples(val_predictions, Y_VAL_MULTI)
	val_dict = dict_compare_tuples(val_comps)
	err, acc = eval_accuracy(val_dict, 5)
	print "Validation error: ", err
	print "Validation accuracy: ", acc

	test_predictions = model.predict(X_TEST)
	test_comps = get_comparison_tuples(test_predictions, Y_TEST_MULTI)
	test_dict = dict_compare_tuples(test_comps)
	err, acc = eval_accuracy(test_dict, 5)
	print "Test error: ", err
	print "Test accuracy: ", acc


