import numpy as np

'''
Return number of classification errors

Arguments pred and labels are both (n, 1) ndarrays.
'''
def binary_classif_err(pred, labels):
    inner_prod = np.multiply(pred, labels)
    inner_prod[inner_prod == 1] = 0 # 1 -> correct
    inner_prod[inner_prod == -1] = 1 # -1 -> error
    errors = np.sum(inner_prod)
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
