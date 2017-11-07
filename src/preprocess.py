import numpy as np
import nltk
import argparse
import pdb
from get_yelp_data import get_review_data

'''
Preprocess reviews from csv file and return 
featurized inputs X and labels Y.

Args:
csv_file (str): path to csv file containing review data

Returns:
(X, Y) tuple where X is (n, d) ndarray and Y is (n, 1) 
ndarray containing +1 or -1.
'''
def preprocess(csv_file):

    review_data = get_review_data(csv_file)
    n, = review_data.shape

    rem_punc = ['.', ',']
    stopwords = set(nltk.corpus.stopwords.words('english'))

    errors = 0
    good_ids = []
    tokens = {}
    pos = {}
    dictionary = {}

    # clean up by tokenizing and tagging parts of speech
    for i in xrange(n):

        row = review_data[i]
        review = row['text']
        review_id = row['review_id']
        
        try:
            # separates words from punctuation
            separated = nltk.word_tokenize(review)
            tokens[review_id] = [token.lower() for token in separated if token not in stopwords if token not in rem_punc]

            # adds unique tokens to dictionary
            for token in tokens[review_id]:
                if token not in dictionary:
                    dictionary[token] = len(dictionary)

            # tag part of speech or punctuation for each separated item
            pos[review_id] = nltk.pos_tag(tokens[review_id])

            # save list of ids of correctly preprocessed reviews
            good_ids.append(review_id)
        
        except:
            print review
            errors += 1
            print "Couldn't tokenize review", review_id

    d = len(dictionary)
    print "total reviews: %d" % n
    print "total errors: %d" % errors
    print "dictionary size: %d" % d

    # X is feature matrix from the bag of words model
    # Y_multi is multi-class labels matrix
    X = np.zeros((n, d))
    Y_multi = np.zeros((n, 1))

    for i in xrange(n):

        row = review_data[i]
        review_id = row['review_id']
        rating = row['stars']

        if review_id in good_ids:
            for token in tokens[review_id]:
                if token in dictionary:
                    X[i][dictionary[token]] = 1

        Y_multi[i] = int(rating)
    
    # create binary star ratings where 1-2 is -1 and 3-5 is +1
    Y_binary = np.where((Y_multi > 2), 1, -1)

    return (X, Y_multi, Y_binary)

if __name__ == "__main__":
    csv_file = 'data/review.csv'
    X, Y_m, Y_b = preprocess(csv_file)
    print "X (feature matrix) is: ", X
    print "Y_m (multi-class labels) is: ", Y_m
    print "Y_b (binary labels) is: ", Y_b



