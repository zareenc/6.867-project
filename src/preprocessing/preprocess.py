import numpy as np
import nltk
import argparse
import pdb
from sklearn.feature_extraction.text import TfidfTransformer
from get_yelp_data import get_review_data, get_business_data, \
    get_user_data, strip_bytes


class Preprocessor:


    REM_PUNC = ['.', ',']
    STOPWORDS = set(nltk.corpus.stopwords.words('english'))
    ATTRIBUTE_NAMES_DISCRETE = ['city']
    ATTRIBUTE_NAMES_CONT = ['average_stars']


    '''
    Initializes Preprocessor object.

    self.attributes_discrete = {'city': ['Toronto', 'Ontario'...]}
    self.attributes_cont = ['average_stars'...]
    self.business_data = {{'business_id':10, 'name':'Garaje', 'city':'San Francisco'...},...}
    self.user_data = {{'user_id':10, 'name':'Sebastien', 'average_stars':3.5...},...}
    '''
    def __init__(self, review_csv_file, business_csv_file='', user_csv_file='', verbose=False):
        self.review_data = get_review_data(review_csv_file)
        self.attributes_discrete, self.attributes_cont = self.init_attributes()
        self.business_data = self.make_business_dict(business_csv_file)
        self.user_data = self.make_user_dict(user_csv_file)
        self.n, = self.review_data.shape
        self.verbose = verbose

        self.errors = 0
        self.good_ids = set()
        self.tokens = {}
        self.pos = {}
        self.words_dictionary = {}

    """Initialize attributes dictionaries"""
    def init_attributes(self):
        attributes_discrete = {}
        for a_name in self.ATTRIBUTE_NAMES_DISCRETE:
            attributes_discrete[a_name] = []
        attributes_cont = self.ATTRIBUTE_NAMES_CONT
        return attributes_discrete, attributes_cont
        
    """
    Returns the desired attribute for a particular review with the
    given review, business, and user id.
    """
    def get_attribute_data(self, review_id, business_id, user_id, attribute):
        if attribute == 'city':
            try:
                return strip_bytes(self.business_data[business_id][attribute])
            except:
                print("Could not find business id %s" % business_id)
                return None
        elif attribute == 'average_stars':
            try:
                return strip_bytes(self.user_data[user_id][attribute])
            except:
                print("Could not find user id %s" % user_id)
                return None

    """
    Make dictionary of user id's to user information.
    Can optionally give a text file of user id's to filter with.
    """  
    def make_user_dict(self, user_csv_file):
        if user_csv_file is None or len(user_csv_file) == 0:
            return {}

        # get user data
        user_data = get_user_data(user_csv_file)

        print('making dictionary of users...')
        user_dict = {}
        for row in user_data:
            # populate user dict
            user_dict[strip_bytes(row['user_id'])] = row

            # populate discrete attributes dict with found values
            for a_name in self.ATTRIBUTE_NAMES_DISCRETE:
                try:
                    a = strip_bytes(row[a_name].title())
                    if a not in self.attributes_discrete[a_name]:
                        self.attributes_discrete[a_name].append(a)
                except:
                    continue
        print('done making dictionary of users')
        return user_dict


    """
    Make dictionary of business id's to business information.
    Can optionally give a text file of business id's to filter with.
    """  
    def make_business_dict(self, business_csv_file):
        if business_csv_file is None or len(business_csv_file) == 0:
            return {}

        # get business data
        business_data = get_business_data(business_csv_file)

        # populate business and attributes dicts
        print('making dictionary of businesses...')
        business_dict = {}
        for row in business_data:
            business_dict[strip_bytes(row['business_id'])] = row
            for a_name in self.ATTRIBUTE_NAMES_DISCRETE:
                try:
                    a = strip_bytes(row[a_name].title())
                    if a not in self.attributes_discrete[a_name]:
                        self.attributes_discrete[a_name].append(a)
                except:
                    continue
        print('done making dictionary of businesses')
        return business_dict


    """Setter for business dictionary"""
    def set_business_data(self, business_data):
        self.business_data = business_data

    """Setter for user dictionary"""
    def set_user_data(self, user_data):
        self.user_data = user_data

    """Setter for attributes"""
    def set_attributes(self, attributes_discrete):
        self.attributes_discrete = attributes_discrete


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
        
        del review_row
        del review
        del review_id

        if self.verbose:
            print("total reviews: %d" % self.n)
            print("total errors: %d" % self.errors)
            print("dictionary size: %d" % len(self.words_dictionary))
            print("Other features:")
            for attribute in self.attributes_discrete:
                print(str(attribute) + " size: %d" % len(self.attributes_discrete[attribute]))
            for attribute in self.attributes_cont:
                print(str(attribute))

        return


    """ featurized inputs X and labels Y """
    def featurize(self, words_dict, multiclass, frequency=False, tf_idf=False, feature_attributes_to_use=[]):
        # X is feature matrix from the bag of words model
        # Y_multi is multi-class labels matrix
        l = len(words_dict)
        X = np.zeros((self.n, l))
        Y_multi = np.zeros((self.n, 1))

        for i in range(self.n):
            review_row = self.review_data[i]
            review_id = strip_bytes(review_row['review_id'])
            rating = review_row['stars']

            if review_id in self.good_ids:
                for token in self.tokens[review_id]:
                    if token in words_dict:
                        if frequency:
                            X[i][words_dict[token]] += 1
                        else:
                            X[i][words_dict[token]] = 1

            Y_multi[i] = int(rating)
        print("created X (feature) of shape", X.shape)

        # delete these variables when done
        del review_row
        del review_id
        del rating

        # normalize frequency counts in featurized inputs
        if frequency and tf_idf:
            print("applying tf_idf transformation to X")
            tfidf_transformer = TfidfTransformer()
            X = tfidf_transformer.fit_transform(X).toarray()
            
        # include other discrete attributes in feature vector
        for attribute in self.attributes_discrete.keys():
            if str(attribute) in feature_attributes_to_use:
                print(str(attribute), "in feature_attributes_to_use for discrete attributes")
                option_list_len = len(self.attributes_discrete[attribute])

                # this is new feature vector that will be concatenated
                print("size of %s is %d" % (str(attribute), option_list_len))
                Xnew = np.zeros((self.n, option_list_len))

                for i in range(self.n):
                    review_row = self.review_data[i]
                    review_id = strip_bytes(review_row['review_id'])
                    business_id = strip_bytes(review_row['business_id'])
                    user_id = strip_bytes(review_row['user_id'])
                        
                    if review_id in self.good_ids:
                        option_list = self.attributes_discrete[attribute]
                        option = self.get_attribute_data(review_id, \
                                 business_id, user_id, attribute)
                        if option is not None:
                            Xnew[i][option_list.index(option)] = 1
                            
                # concatenate this
                X = np.hstack((X, Xnew))
                print("new X matrix is: ", X)

        # include other continuous attributes in feature vector
        for attribute in self.attributes_cont:
            if str(attribute) in feature_attributes_to_use:
                print(str(attribute), "in feature_attributes_to_use for continuous attributes")
                
                # this is new 1-dimensional feature vector that will be concatenated
                print("size of %s is %d" % (str(attribute), 1))
                Xnew = np.zeros((self.n, 1))

                for i in range(self.n):
                    review_row = self.review_data[i]
                    review_id = strip_bytes(review_row['review_id'])
                    business_id = strip_bytes(review_row['business_id'])
                    user_id = strip_bytes(review_row['user_id'])

                    if review_id in self.good_ids:
                        option = self.get_attribute_data(review_id, \
                                 business_id, user_id, attribute)
                        if option is not None:
                            Xnew[i][0] = option

                # concatenate this
                X = np.hstack((X, Xnew))
                print("new X matrix is: ", X)

        # delete these variables when done
        # del review_row
        # del review_id
        # del business_id

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
            '--user_csv_file',
            type=str,
            help='The user csv file.',
            )
    parser.add_argument(
            '--multi_class',
            action='store_true',
            default=False,
            required=False,
            help='multiclass or binary classification',
            )
    parser.add_argument(
            '--frequency',
            action='store_true',
            default=False,
            required=False,
            help='use frequency of presence for bag of words featurization',
            )
    parser.add_argument(
            '--tf_idf',
            action='store_true',
            default=False,
            required=False,
            help='use tf_idf normalization for bag of words featurization',
            )

    args = parser.parse_args()
    review_csv_file = args.review_csv_file
    business_csv_file = args.business_csv_file
    user_csv_file = args.user_csv_file
    multi_class = args.multi_class
    tf_idf = args.tf_idf
    frequency = args.frequency
    feature_attributes_to_use = ['city', 'average_stars']

    preprocess = Preprocessor(review_csv_file, business_csv_file, user_csv_file)

    print('cleaning up reviews...')
    preprocess.cleanup(modify_words_dictionary=True)
    print('making words dictionary...')
    dic = preprocess.get_words_dictionary()
    print('featurizing reviews...')
    X, Y = preprocess.featurize(dic, multi_class, \
           tf_idf=tf_idf, frequency=frequency, \
           feature_attributes_to_use=feature_attributes_to_use)

    print("X (feature matrix) shape is: ", X.shape)
    print("Y (labels) shape is: ", Y.shape)

