import numpy as np
import nltk
import argparse
import pdb
from sklearn.feature_extraction.text import TfidfTransformer
from get_yelp_data import get_review_data, get_business_data, get_filtered_business_data


class Preprocessor:


    REM_PUNC = ['.', ',']
    STOPWORDS = set(nltk.corpus.stopwords.words('english'))


    def __init__(self, review_csv_file, business_csv_file='', business_filter_file='', verbose=False):
        self.review_data = get_review_data(review_csv_file)
        self.business_data = self.make_business_dict(business_csv_file, business_filter_file)
        self.n, = self.review_data.shape
        self.d = None   # later set to dictionary size
        self.verbose = verbose

        self.errors = 0
        self.good_ids = set()
        self.tokens = {}
        self.pos = {}
        self.dictionary = {}


    """
    Make dictionary of business id's to business information.
    Can optionally give a text file of business id's to filter with.
    """   
    def make_business_dict(self, business_csv_file, business_filter_file=''):
        print('making dictionary of businesses...')
        business_data = get_business_data(business_csv_file)
        business_dict = {}
        for row in business_data:
            business_dict[row['business_id']] = row
        return business_dict


    """Clean up reviews from csv file and . """
    def cleanup(self, lower=True, remove_stopwords=True, stem=True):
        # clean up by tokenizing and tagging parts of speech
        for i in range(self.n):

            row = self.review_data[i]
            review = row['text'].decode("utf-8")
            review_id = row['review_id'].decode("utf-8")

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
                    print("Couldn't tokenize review", review_id)
        del row
        del review
        del review_id

        self.d = len(self.dictionary)
        if self.verbose:
            print("total reviews: %d" % self.n)
            print("total errors: %d" % self.errors)
            print("dictionary size: %d" % self.d)

        return


    """ featurized inputs X and labels Y """
    def featurize(self, some_dictionary, frequency=False, tf_idf=False):
        # X is feature matrix from the bag of words model
        # Y_multi is multi-class labels matrix
        l = len(some_dictionary)
        X = np.zeros((self.n, l))
        Y_multi = np.zeros((self.n, 1))

        for i in range(self.n):

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

        # normalize frequency counts in featurized inputs
        if frequency and tf_idf:
            tfidf_transformer = TfidfTransformer()
            X = tfidf_transformer.fit_transform(X).toarray()
        
        del row
        del review_id
        del rating
        # Y_binary is binary labels matrix
        # binary star ratings where 1-2 is -1 and 3-5 is +1
        Y_binary = np.where((Y_multi > 2), 1, -1)
        # Y_binary = np.where((Y_multi > 3), 1, -1)

        return (X, Y_multi, Y_binary)


    """ return the dictionary obtained from preprocessing """
    def get_dictionary(self):
        return self.dictionary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Preprocess CSV file and return featured reviews (X) and corresponding labels (Y).',
            )

    parser.add_argument(
            'review_csv_file',
            type=str,
            help='The review csv file to featurize.',
            )
    parser.add_argument(
            '--business_csv_file',
            type=str,
            help='The business csv file.',
            )
    parser.add_argument(
            '--business_filter_file',
            type=str,
            help='Text file of business id\'s to filter business with.',
            )

    args = parser.parse_args()
    review_csv_file = args.review_csv_file
    business_csv_file = args.business_csv_file
    business_filter_file = args.business_filter_file
    preprocess = Preprocessor(review_csv_file, business_csv_file, business_filter_file)

    print('cleaning up reviews...')
    preprocess.cleanup()
    print('making dictionary...')
    dic = preprocess.get_dictionary()
    print('featurizing reviews...')
    X, Y_m, Y_b = preprocess.featurize(dic)

    print("X (feature matrix) is: ", X)
    print("Y_m (multi-class labels) is: ", Y_m)
    print("Y_b (binary labels) is: ", Y_b)

