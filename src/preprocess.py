import numpy as np
import nltk
import argparse
import pdb
from get_yelp_data import get_review_data

'''
Preprocess reviews from csv file and return 
inputs X and labels Y.

Args:
csv_file (str): path to csv file containing review data

Returns:
(X, Y) tuple where X is (n, d) ndarray and Y is (n, 1) 
ndarray containing +1 or -1.
'''
def preprocess(csv_file):

    review_data = get_review_data(csv_file)
    n, = review_data.shape
    X = np.asarray((n,1))
    Y = np.asarray((n, 1))

    errors = 0
    good_ids = []
    tokens = {}
    pos = {}

    # tokenize and tag parts of speech
    for i in xrange(n):
        row = review_data[i]
        review = row['text']
        id = row['review_id']
        try:
            tokens[id] = nltk.word_tokenize(review)
            pos[id] = nltk.pos_tag(tokens[id])
            good_ids.append(id)
        except:
            print review
            errors += 1
            print "Couldn't tokenize review", id

    print "total reviews: %d" % n
    print "total errors: %d" % errors
    return (X, Y)

if __name__ == "__main__":
    csv_file = 'data/review.csv'
    X, Y = preprocess(csv_file)
