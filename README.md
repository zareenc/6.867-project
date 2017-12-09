# 6.867 Final Project

## Requirements
- Python 3
- Python packages:
   - `numpy`
   - `scikit-learn`
   - `statistics`
- NLTK packages:
```
$ sudo pip install -U nltk
$ python
> import nltk
> nltk.download('punkt')
> nltk.download('averaged_perceptron_tagger')
> nltk.download('stopwords')
```

## Using this code

See the src/preprocessing and src/classifiers for more detailed information about processing data and running classifiers. While each script to process the data takes in a different set of arguments, most of the classifiers take in the same ones.

The usual process to run a classifier is
```
python classifier.py path_to_train_csv path_to_validation_csv path_to_test_csv
```

Some of the classifiers have additional arguments. See the README in the src/classifiers directory for more details.
