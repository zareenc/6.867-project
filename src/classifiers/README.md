## Classifiers
Below are examples of calling the various classification methods provided on the training, validation, and test data.

### Perceptron
```
python perceptron.py train_data.csv val_data.csv test_data.csv
```

### Logistic Regression
Binary classification:
```
python logistic_regression.py train_data.csv val_data.csv test_data.csv
```

Multiclass classification:
```
python logistic_regression.py train_data.csv val_data.csv test_data.csv --multi_class=True
```

### SVM
By default, SVM runs both binary and multiclass classification with linear and RBF kernels:
```
python svm.py train_data.csv val_data.csv test_data.csv
```

Specific kernels or types of classification can be turned off by setting any combination of the following flags:
```
--norun_bin_lin            Don't run binary classification with linear kernel
--norun_bin_rbf            Don't run binary classification with RBF kernel
--norun_multi_lin          Don't run multiclass classification with linear kernel
--norun_multi_rbf          Don't run multiclass classification with RBF kernel
```

For example, you can run only binary classification as follows:
```
python svm.py train_data.csv val_data.csv test_data.csv --norun_multi_lin --norun_multi_rbf
```

### K Means
```
python kmeans.py train_data.csv val_data.csv test_data.csv
```
