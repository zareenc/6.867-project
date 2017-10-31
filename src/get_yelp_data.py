from numpy import genfromtxt
import numpy as np
import argparse
import pdb

def get_review_data(csv_file):
    names = ('funny', 'user_id', 'review_id', 'text', 'business_id', \
                 'stars', 'date', 'useful', 'cool')
    filling = None

    print "getting data"
    data = genfromtxt(csv_file, dtype=None, names=names, \
                             delimiter='\t', skip_header=1, comments=None, \
                             filling_values=filling)
    print "done getting data"

def get_business_data(csv_file):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Get Yelp data.',
            )
    parser.add_argument(
            'type',
            type=str,
            help='The csv file to load.',
            )

    args = parser.parse_args()
    type = args.type

    csv_file = 'data/%s.csv' % type

    if type == 'review':
        get_review_data(csv_file)
    elif type == 'business':
        get_business_data(csv_file)
