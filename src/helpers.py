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
