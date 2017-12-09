import numpy as np
import nltk
import argparse
import pdb
from sklearn.feature_extraction.text import TfidfTransformer
from get_yelp_data import get_review_data, get_business_data, get_filtered_business_data


class Preprocessor:


    REM_PUNC = ['.', ',']
    STOPWORDS = set(nltk.corpus.stopwords.words('english'))
    ATTRIBUTE_NAMES = ['city']


    def __init__(self, review_csv_file, business_csv_file='', business_filter_file='', verbose=False):
        self.review_data = get_review_data(review_csv_file)
        self.business_data, self.attributes = self.make_business_dict(business_csv_file, business_filter_file)
        self.n, = self.review_data.shape
        self.verbose = verbose

        self.errors = 0
        self.good_ids = set()
        self.tokens = {}
        self.pos = {}
        self.words_dictionary = {}


    """
    Make dictionary of business id's to business information.
    Can optionally give a text file of business id's to filter with.
    """  
    def make_business_dict(self, business_csv_file, business_filter_file=''):
        if len(business_csv_file) == 0:
            return {}, {}

        print('making dictionary of businesses...')
        business_data = get_business_data(business_csv_file)
        if len(business_filter_file) > 0:
            label, bus_ids = construct_filtered_set(business_filter_file)
            business_data = get_filtered_business_data(business_data, BUSINESS_BUSID_IDX, bus_ids)

        # initialize attributes dict
        attributes = {}
        for a_name in self.ATTRIBUTE_NAMES:
            attributes[a_name] = []

        # populate business and attributes dicts
        business_dict = {}
        for row in business_data:
            business_dict[row['business_id']] = row
            for a_name in self.ATTRIBUTE_NAMES:
                a = row[a_name].title()
                if a not in attributes[a_name]:
                    attributes[a_name].append(a)

        return business_dict, attributes


    """Setter for business dictionary"""
    def set_business_dict(self, business_dict):
        self.business_dict = business_dict


    """Setter for attributes"""
    def set_attributes(self, attributes):
        self.attributes = attributes


    """Clean up reviews from csv file and . """
    def cleanup(self, lower=True, remove_stopwords=True, stem=True, modify_words_dictionary=False):
        # clean up by tokenizing and tagging parts of speech
        for i in range(self.n):

            review_row = self.review_data[i]
            review = review_row['text'].decode("utf-8")
            review_id = review_row['review_id'].decode("utf-8")

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

                # adds unique tokens to words dictionary
                if modify_words_dictionary:
                    for token in self.tokens[review_id]:
                        if token not in self.words_dictionary:
                            self.words_dictionary[token] = len(self.words_dictionary)

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

        if self.verbose:
            print("total reviews: %d" % self.n)
            print("total errors: %d" % self.errors)
            print("dictionary size: %d" % len(self.words_dictionary))
            print("Other features:")
            for attribute in self.attributes:
                print(str(attribute) + " size: %d" % len(self.attributes[attribute]))

        return


    """ featurized inputs X and labels Y """
    def featurize(self, words_dict, multiclass=False, frequency=False, tf_idf=False, feature_attributes_to_use=[]):
        # X is feature matrix from the bag of words model
        # Y_multi is multi-class labels matrix
        l = len(words_dict)
        X = np.zeros((self.n, l))
        Y_multi = np.zeros((self.n, 1))

        for i in range(self.n):

            review_row = self.review_data[i]
            review_id = review_row['review_id'].decode("utf-8")
            rating = review_row['stars'].decode("utf-8")

            if review_id in self.good_ids:
                for token in self.tokens[review_id]:
                    if token in words_dict:
                        if frequency:
                            X[i][words_dict[token]] += 1
                        else:
                            X[i][words_dict[token]] = 1

            Y_multi[i] = int(rating)

        # delete these variables when done
        del review_row
        del review_id
        del rating

        # normalize frequency counts in featurized inputs
        if frequency and tf_idf:
            tfidf_transformer = TfidfTransformer()
            X = tfidf_transformer.fit_transform(X).toarray()

            
        # include other attributes in feature vector
        for attribute in self.attributes.keys(): 
            if str(attribute) in feature_attributes_to_use:
                option_list_len = len(self.attributes[attribute])

                # this is new feature vector that will be concatenated
                Xnew = np.zeros(self.n, option_list_len)

                    for i in range(self.n):
                        review_row = self.review_data[i]
                        review_id = review_row['review_id'].decode("utf-8")
                        business_id = review_row['business_id'].decode("utf-8")

                        if review_id in self.good_ids:
                            option_list = self.attributes[attribute]
                            option = self.business_data[business_id][attribute]
                            Xnew[i][option_list.index(option)] = 1

            # concatenate this
            X = np.hstack((X, Xnew))
            print("new X matrix is: ", X)

        # delete these variables when done
        del review_row
        del review_id
        del business_id

        print("final X matrix is: ", X)

        # delete these variables when done
        del review_row
        del review_id
        del rating

        # Y_binary is binary labels matrix
        # binary star ratings where 1-2 is -1 and 3-5 is +1
        Y_binary = np.where((Y_multi > 2), 1, -1)
        # Y_binary = np.where((Y_multi > 3), 1, -1)

        if multiclass:
            return (X, Y_multi)

        return (X, Y_binary)


    """ return the words dictionary obtained from preprocessing """
    def get_words_dictionary(self):
        return self.words_dictionary


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
    print('making words dictionary...')
    dic = preprocess.get_words_dictionary()
    print('featurizing reviews...')
    X, Y = preprocess.featurize(dic)

    print("X (feature matrix) is: ", X)
    print("Y (labels) is: ", Y)

