import numpy as np
import nltk
import argparse
import pdb
from sklearn.feature_extraction.text import TfidfTransformer
from get_yelp_data import get_review_data


class Preprocessor:


    REM_PUNC = ['.', ',']
    STOPWORDS = set(nltk.corpus.stopwords.words('english'))


    def __init__(self, csv_file, verbose=False):
        self.review_data = get_review_data(csv_file)
        self.n, = self.review_data.shape
        self.d = None   # later set to words dictionary size
        self.verbose = verbose

        self.errors = 0
        self.good_ids = set()
        self.tokens = {}
        self.pos = {}
        self.words_dictionary = {}
        self.city_dictionary = {}


    """Clean up reviews from csv file and . """
    def cleanup(self, lower=True, remove_stopwords=True, stem=True, use_city=False):
        # clean up by tokenizing and tagging parts of speech
        for i in xrange(self.n):

            review_row = self.review_data[i]
            review = review_row['text']
            review_id = review_row['review_id']
            
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
                    print "Couldn't tokenize review", review_id

        self.d = len(self.words_dictionary)

        # loop over again, using only good ids
        for i in xrange(self.n):

            review_row = self.review_data[i]
            review_id = review_row['review_id']
            business_id = review_row['business_id']

            current_feature_vector_length

            if use_city:
                city = self.business_data[business_id]['city']
                if city not in self.city_dictionary:
                    self.city_dictionary[city] = len(self.city_dictionary) + self.d


        if self.verbose:
            print "total reviews: %d" % self.n
            print "total errors: %d" % self.errors
            print "words dictionary size: %d" % self.d
            print "Other features:"
            print "city dictionary size: %d" % len(self.city_dictionary)

        return


    """ featurized inputs X and labels Y """
    def featurize(self, words_dict, frequency=False, tf_idf=False):
        # X is feature matrix from the bag of words model
        # Y_multi is multi-class labels matrix
        l = len(words_dict)
        X = np.zeros((self.n, l))
        Y_multi = np.zeros((self.n, 1))

        for i in xrange(self.n):

            review_row = self.review_data[i]
            review_id = review_row['review_id']
            rating = review_row['stars']
            business_id = review_row['business_id']

            if review_id in self.good_ids:
                for token in self.tokens[review_id]:
                    if token in words_dict:
                        if frequency:
                            X[i][words_dict[token]] += 1
                        else:
                            X[i][words_dict[token]] = 1

            Y_multi[i] = int(rating)

        # normalize frequency counts in featurized inputs
        if frequency and tf_idf:
            tfidf_transformer = TfidfTransformer()
            X = tfidf_transformer.fit_transform(X).toarray()
        
        # concatenate other features below, if flags set

        # include city data
        if len(city_dictionary) > 0:
            city = self.business_data[business_id]['city']

        # include other data...

        # Y_binary is binary labels matrix
        # binary star ratings where 1-2 is -1 and 3-5 is +1
        Y_binary = np.where((Y_multi > 2), 1, -1)

        return (X, Y_multi, Y_binary)


    """ return the words dictionary obtained from preprocessing """
    def get_words_dictionary(self):
        return self.words_dictionary


    """ return the words dictionary obtained from preprocessing """
    def get_city_dictionary(self):
        return self.city_dictionary


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
    dic = preprocess.get_words_dictionary()
    X, Y_m, Y_b = preprocess.featurize(dic)

    print "X (feature matrix) is: ", X
    print "Y_m (multi-class labels) is: ", Y_m
    print "Y_b (binary labels) is: ", Y_b

