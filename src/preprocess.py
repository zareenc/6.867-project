import numpy as np
import nltk
import argparse
import pdb
from get_yelp_data import get_review_data


def preprocess(csv_file):
    review_data = get_review_data(csv_file)
    n, = review_data.shape

    errors = 0
    good_ids = []
    tokens = {}
    pos = {}
    # tokenize and tag parts of speech
    for i in xrange(n):
        row = review_data[i]
        review = row['text']
        try:
            id = row['review_id']
            tokens[id] = nltk.word_tokenize(review)
            pos[id] = nltk.pos_tag(tokens[id])
            good_ids.append(id)
            pdb.set_trace()
        except:
            print review
            errors += 1
            print "Couldn't tokenize"

    print "total reviews: %d" % n
    print "total errors: %d" % errors

if __name__ == "__main__":
    csv_file = 'data/review.csv'
    preprocess(csv_file)
