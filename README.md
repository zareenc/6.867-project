# 6.867 Final Project

## Requirements
- Python 2.7
- Numpy
- NLTK packages:
```
$ sudo pip install -U nltk
$ python
> import nltk
> nltk.download('punkt')
> nltk.download('averaged_perceptron_tagger')
> nltk.download('stopwords')
```

## Examples
Load review data:
```
from get_yelp_data import *
data = get_review_data('../data/review.csv')
```

Get features and labels from review data:
```
from preprocess import *

# training data
preprocessor_train = Preprocessor('../data/review_train.csv')
preprocessor_train.cleanup()
dict = preprocessor_train.get_dictionary()
X_train, Y_train_multi, Y_train_binary = preprocessor_train.featurize(dict)

# test data
preprocessor_test = Preprocessor('../data/review_test.csv')
preprocessor_test.cleanup()
X_test, Y_test_multi, Y_test_binary = preprocessor_test.featurize(dict)
```

## Parsing the Original JSON Data
In order to convert one of the original JSON files from the dataset into a csv file, run
```
python json_to_csv_converter.py path_to_json_file path_to_new_csv_file
```

If you want to filter the business.json file and get a list of business IDs based on some criteria, run the file with the name of a text file to store the filtered IDs.
```
python json_to_csv_converter.py path_to_business_json path_to_new_business_csv_file --filtered_txt_file=path_to_new_txt_file
```
This will generate a list of IDs according to the criteria in json_to_csv_converter.py. There are two supported types of criteria to filter by: single-valued and categories.
  * Single valued criteria are fields such as "state" and "city". They are fields that have a single name and single value.
  * Categories are strings in the "categories" list field in the original JSON file.

To filter by single-valued field, the name of the field and desired value must be in the `scalar_filter_columns` and `scalar_filter_values` lists at the same index. To filter by a category, add the exact category name to the `filter_categories` list. These filtering criteria can be run in any order.

## Filtering the CSV Data
In order to filter a generated review.csv file by any filter categories, call
```
python get_yelp_data.py review path_to_review_csv path_to_filter_txt path_to_new_csv_file
```
This will create a new csv file of reviews with the criteria found in the given filter.txt file. Right now it expects the txt file to look like
```
business_id
id_1
id_2
...
```
If the given txt file does not start with "business_id", a new file won't be generated.
