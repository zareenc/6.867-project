import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../preprocessing')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
import statistics
import pdb
from helpers import *
from preprocess import *

CLUSTER_NUMBER = 5

def get_comparison_tuples(predictions, labels_multi):
	compares = []
	for i in range(len(labels_multi)):
		compares.append((int(labels_multi[i]), predictions[i]))
	return compares

def dict_compare_tuples(compares):
	dict_compares = {}
	for i in range(len(compares)):
		if compares[i] not in dict_compares:
			dict_compares[compares[i]] = 1
		else:
			dict_compares[compares[i]] += 1
	print("dict compares is:", dict_compares)
	return dict_compares

def cluster_stats(compares):
	mean = {}
	stdev = {}
	median = {}
	mode = {}
	#initialize all to None
	for i in range(CLUSTER_NUMBER):
		mean[i] = None
		stdev[i] = None
		median[i] = None
		mode[i] = None

	listoflists = []
	for i in range(CLUSTER_NUMBER):
		clusterlist = []
		for tup in compares:
			pred = tup[0]
			cluster = tup[1]
			if cluster == i:
				clusterlist.append(pred)
		listoflists.append(clusterlist)

	for i in range(CLUSTER_NUMBER):
		try:
			mean[i] = statistics.mean(listoflists[i])
		except:
			print("list missing data points")
		try:
			stdev[i] = statistics.stdev(listoflists[i])
		except:
			print("list missing data points")
		try:
			median[i] = statistics.median(listoflists[i])
		except:
			print("list missing data points")
		try:
			mode[i] = statistics.mode(listoflists[i])
		except:
			print("list missing data points")
	
	print(listoflists)
	return mean, stdev, median, mode



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
	multi_class = True				# K-means should always be multiclass

	train_csv = args.train_file
	val_csv = args.val_file
	test_csv = args.test_file

	pre_train = Preprocessor(train_csv, args.business_csv)
	pre_val = Preprocessor(val_csv, args.business_csv)
	pre_test = Preprocessor(test_csv, args.business_csv)

	features = ['city']

	pre_train.cleanup(modify_words_dictionary=True)
	dict_train = pre_train.get_words_dictionary()
	X_TRAIN, Y_TRAIN_MULTI = pre_train.featurize(dict_train, multi_class, feature_attributes_to_use=features)

	pre_val.cleanup()
	X_VAL, Y_VAL_MULTI = pre_val.featurize(dict_train, multi_class, feature_attributes_to_use=features)

	pre_test.cleanup()
	X_TEST, Y_TEST_MULTI = pre_test.featurize(dict_train, multi_class, feature_attributes_to_use=features)
	
	# training
	model = SpectralClustering(n_clusters=CLUSTER_NUMBER)
	model.fit(X_TRAIN)


	train_predictions = model.fit_predict(X_TRAIN)
	train_comps = get_comparison_tuples(train_predictions, Y_TRAIN_MULTI)
	tr_mean, tr_stdev, tr_median, tr_mode = cluster_stats(train_comps)
	print("Training cluster mean: ", tr_mean)
	print("Training cluster standard deviation: ", tr_stdev)
	print("Training cluster median: ", tr_median)
	print("Training cluster mode: ", tr_mode)
	train_dict = dict_compare_tuples(train_comps)


	val_predictions = model.fit_predict(X_VAL)
	val_comps = get_comparison_tuples(val_predictions, Y_VAL_MULTI)
	v_mean, v_stdev, v_median, v_mode = cluster_stats(val_comps)
	print("Validation assignment mean: ", v_mean)
	print("Validation assignment standard deviation: ", v_stdev)
	print("Validation assignment median: ", v_median)
	print("Validation assignment mode: ", v_mode)
	val_dict = dict_compare_tuples(val_comps)


	test_predictions = model.fit_predict(X_TEST)
	test_comps = get_comparison_tuples(test_predictions, Y_TEST_MULTI)
	te_mean, te_stdev, te_median, te_mode = cluster_stats(test_comps)
	print("Test assignment mean: ", te_mean)
	print("Test assignment standard deviation: ", te_stdev)
	print("Test assignment median: ", te_median)
	print("Test assignment mode: ", te_mode)
	test_dict = dict_compare_tuples(test_comps)



