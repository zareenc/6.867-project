import numpy as np
import argparse

''' Generic parser for classifiers '''
def ClassificationParser():
    parser = argparse.ArgumentParser(
            description='Classification Parser',
            )
    parser.add_argument(
            'train_file',
            type=str,
            help='csv file with training data',
            )
    parser.add_argument(
            'val_file',
            type=str,
            help='csv file with validation data',
            )
    parser.add_argument(
            'test_file',
            type=str,
            help='csv file with test data',
            )
    parser.add_argument(
            '--multi_class',
            type=bool,
            default=False,
            required=False,
            help='multiclass or binary classification',
            )
    parser.add_argument(
            '--frequency',
            type=bool,
            default=False,
            required=False,
            help='use frequency of presence for bag of words featurization',
            )
    parser.add_argument(
            '--tf_idf',
            type=bool,
            default=False,
            required=False,
            help='use tf_idf normalization for bag of words featurization',
            )
    return parser

'''
Return number of classification errors

Arguments pred and labels are both (num_samples, 1) ndarrays.
'''
def classif_err(pred, labels):
    errors = len(pred) - np.count_nonzero(pred == labels)
    return float(errors), float(len(pred) - errors) / len(pred)


'''
Append classification results to file.

Arguments:
   results_file: filepath of file to write to
   params_list: list of dicts containing parameters used {str: *}
   results_list: list of dict containing results {str: *}

Example:
   params_list = [{'gamma':0.1}, {'gamma':0.01}]
   results_list = [{'train':0.99,'test':0.80}, {'train':0.99,'test':0.82}]
   write_results('results.txt', params_list, results_list)
'''
def write_results(results_file, params_list, results_list):

    assert len(params_list) == len(results_list)

    with open(results_file, 'a') as f:
        for i in xrange(len(params_list)):
            results = results_list[i]
            params = params_list[i]

            params_str = ''
            for k, v in params.iteritems():
                params_str += '%s: %s ' % (k, str(v))
            f.write(params_str + '\n')
        
            results_str = ''
            for k, v in results.iteritems():
                results_str += '%s: %s ' % (k, str(v))
            f.write(results_str + '\n')
            f.write('\n')

'''
Turns an (n,) numpy array into an (n, 1) numpy array

Arguments:
   arr: the original (n,) numpy array

Returns arr as an (n, 1) numpy array, does not mutate arr
'''
def expand(arr):
    return np.reshape(arr, (arr.shape[0], 1))
