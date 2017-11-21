import numpy as np
import nltk
import argparse
import pdb
from get_yelp_data import get_review_data


class Preprocessor:


    REM_PUNC = ['.', ',']
    STOPWORDS = set(nltk.corpus.stopwords.words('english'))


    def __init__(self, csv_file, verbose=False):
        self.review_data = get_review_data(csv_file)
        self.n, = self.review_data.shape
        self.d = None   # later set to dictionary size
        self.verbose = verbose

        self.errors = 0
        self.good_ids = set()
        self.tokens = {}
        self.pos = {}
        self.dictionary = {}


    """Clean up reviews from csv file and . """
    def cleanup(self, lower=True, remove_stopwords=True, stem=True):
        # clean up by tokenizing and tagging parts of speech
        for i in xrange(self.n):

            row = self.review_data[i]
            review = row['text']
            review_id = row['review_id']
            
            try:
                # separates words from punctuation
                separated = nltk.word_tokenize(review)

                # make lowercase if lower=True
                if lower:
                    self.tokens[review_id] = [token.lower() for token in separated if token not in self.REM_PUNC]
                else:
                    self.tokens[review_id] = [token for token in separated if token not in self.REM_PUNC]

                # remove stopwords if remove_stopwords=True
                if remove_stopwords:
                    current_tokens = self.tokens[review_id]
                    self.tokens[review_id] = [token for token in current_tokens if token not in self.STOPWORDS]

                # stems words if stem=True
                if stem:
                    current_tokens = self.tokens[review_id]
                    stemmer = nltk.stem.snowball.SnowballStemmer("english")
                    stemmed_tokens = [stemmer.stem(token) for token in current_tokens]
                    self.tokens[review_id] = [token for token in stemmed_tokens]

                # adds unique tokens to dictionary
                for token in self.tokens[review_id]:
                    if token not in self.dictionary:
                        self.dictionary[token] = len(self.dictionary)

                # tag part of speech or punctuation for each separated item
                self.pos[review_id] = nltk.pos_tag(self.tokens[review_id])

                # save list of ids of correctly preprocessed reviews
                self.good_ids.add(review_id)
            
            except:
                self.errors += 1
                if self.verbose:
                    print "Couldn't tokenize review", review_id

        self.d = len(self.dictionary)
        if self.verbose:
            print "total reviews: %d" % self.n
            print "total errors: %d" % self.errors
            print "dictionary size: %d" % self.d

        return


    """ featurized inputs X and labels Y """
    def featurize(self, some_dictionary, frequency=False):
        # X is feature matrix from the bag of words model
        # Y_multi is multi-class labels matrix
        l = len(some_dictionary)
        X = np.zeros((self.n, l))
        Y_multi = np.zeros((self.n, 1))

        for i in xrange(self.n):

            row = self.review_data[i]
            review_id = row['review_id']
            rating = row['stars']

            if review_id in self.good_ids:
                for token in self.tokens[review_id]:
                    if token in some_dictionary:
                        if frequency:
                            X[i][some_dictionary[token]] += 1
                        else:
                            X[i][some_dictionary[token]] = 1

            Y_multi[i] = int(rating)
        
        # Y_binary is binary labels matrix
        # binary star ratings where 1-2 is -1 and 3-5 is +1
        Y_binary = np.where((Y_multi > 2), 1, -1)

        return (X, Y_multi, Y_binary)


    """ return the dictionary obtained from preprocessing """
    def get_dictionary(self):
        return self.dictionary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Preprocess CSV file and return featured reviews (X) and corresponding labels (Y).',
            )

    parser.add_argument(
            'csv_file',
            type=str,
            help='The csv file to featurize.',
            )

    args = parser.parse_args()
    csv_file = args.csv_file
    preprocess = Preprocessor(csv_file)

    preprocess.cleanup()
    dic = preprocess.get_dictionary()
    X, Y_m, Y_b = preprocess.featurize(dic)

    print "X (feature matrix) is: ", X
    print "Y_m (multi-class labels) is: ", Y_m
    print "Y_b (binary labels) is: ", Y_b

