import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../preprocessing')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means_
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import statistics
import pdb
from helpers import *
from preprocess import *

CLUSTER_NUMBER = 5

def train_kmeans(X_train):
	
	# Manually override euclidean to use cosine similarity
	def euc_dist(X, Y = None, Y_norm_squared = None, squared = False):
		#return pairwise_distances(X, Y, metric = 'cosine', n_jobs = 10)
		return np.arccos(cosine_similarity(X, Y))/np.pi

	k_means_.euclidean_distances = euc_dist

	#kmeans = KMeans(n_clusters=CLUSTER_NUMBER, init='k-means++', n_init=500, max_iter=1000, tol=0.0001)
	kmeans = MiniBatchKMeans(n_clusters=CLUSTER_NUMBER, init='k-means++', n_init=500, max_iter=1000, tol=0.000001)
	kmeans.fit(X_train)
	print(kmeans.labels_)
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
			dict_compares[compares[i]] = 1
		else:
			dict_compares[compares[i]] += 1
	print("dict compares is:", dict_compares)
	return dict_compares

def kmeans_stats(compares):
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

'''Not sure if this is a valid way to consider k-means accuracy'''
def eval_accuracy(dict_compares, classes):
	best_pairs = {}
	for i in range(1, classes+1):
		best_pairs[i] = 2*classes
	print("best pairs initialized to: ", best_pairs)

	for i in range(1, classes+1):
		for tup in dict_compares:
			if tup[0] == i:
				if best_pairs[i] == 2*classes:
					best_pairs[i] = tup[1]
				elif dict_compares[(i, tup[1])] > dict_compares[(i, best_pairs[i])]:
					best_pairs[i] = tup[1]
	print("FINAL BEST PAIRS: ", best_pairs)
	
	best_pairs_set = set()
	for pred in best_pairs:
		best_pairs_set.add((pred, best_pairs[pred]))
	print("BEST PAIRS SET: ", best_pairs_set)

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

	pre_train = Preprocessor(train_csv)
	pre_val = Preprocessor(val_csv)
	pre_test = Preprocessor(test_csv)

	pre_train.cleanup(modify_words_dictionary=True)
	dict_train = pre_train.get_words_dictionary()
	X_TRAIN, Y_TRAIN_MULTI = pre_train.featurize(dict_train, multi_class)

	pre_val.cleanup()
	X_VAL, Y_VAL_MULTI = pre_val.featurize(dict_train, multi_class)

	pre_test.cleanup()
	X_TEST, Y_TEST_MULTI = pre_test.featurize(dict_train, multi_class)
	
	# training
	model = train_kmeans(X_TRAIN)

	train_predictions = model.predict(X_TRAIN)
	train_comps = get_comparison_tuples(train_predictions, Y_TRAIN_MULTI)
	tr_mean, tr_stdev, tr_median, tr_mode = kmeans_stats(train_comps)
	print("Training cluster mean: ", tr_mean)
	print("Training cluster standard deviation: ", tr_stdev)
	print("Training cluster median: ", tr_median)
	print("Training cluster mode: ", tr_mode)
	train_dict = dict_compare_tuples(train_comps)
	#err, acc = eval_accuracy(train_dict, 5)
	#print("Training error: ", err)
	#print("Training accuracy: ", acc)

	val_predictions = model.predict(X_VAL)
	val_comps = get_comparison_tuples(val_predictions, Y_VAL_MULTI)
	v_mean, v_stdev, v_median, v_mode = kmeans_stats(val_comps)
	print("Validation assignment mean: ", v_mean)
	print("Validation assignment standard deviation: ", v_stdev)
	print("Validation assignment median: ", v_median)
	print("Validation assignment mode: ", v_mode)
	val_dict = dict_compare_tuples(val_comps)
	#err, acc = eval_accuracy(val_dict, 5)
	#print("Validation error: ", err)
	#print("Validation accuracy: ", acc)

	test_predictions = model.predict(X_TEST)
	test_comps = get_comparison_tuples(test_predictions, Y_TEST_MULTI)
	te_mean, te_stdev, te_median, te_mode = kmeans_stats(test_comps)
	print("Test assignment mean: ", te_mean)
	print("Test assignment standard deviation: ", te_stdev)
	print("Test assignment median: ", te_median)
	print("Test assignment mode: ", te_mode)
	test_dict = dict_compare_tuples(test_comps)
	#err, acc = eval_accuracy(test_dict, 5)
	#print("Test error: ", err)
	#print("Test accuracy: ", acc)


