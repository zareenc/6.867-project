# Data Manipulation and Preprocessing
This directory contains scripts for manipulating the Yelp data from the original JSON files into csv files that can be used by our classifiers. Make sure that you run all of these scripts using Python 3.

## Parsing JSON Data
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

## Filtering CSV Data
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

## Creating Evenly Distributed Datasets
In order to create datasets that have an even distribution of positive and negative reviews or an even distribution of each star rating (1-5) of reviews, call
```
python create_even_data.py path_to_full_review_csv path_to_new_review_csv <number of classes> <number of reviews per class>
```
The number of classes must be 2 or 5. For example, if you want to create a dataset of 1000 reviews of multiclass data, you can call
```
python create_even_data.py data/reviews.csv data/evenly_distributed_reviews.csv 5 200
```
This will create a new csv file of 1000 reviews with the 5 star ratings evenly split across the dataset. This file is made sequentially from the original review file, so it will not be sorted by star rating. To further randomize the data, see the section below.

## Shuffling Data
If you want to further shuffle any csv file, you can call
```
python shuffle_csv.py path_to_original_csv path_to_new_shuffled_csv
```
This will randomly shuffle the entries of the original csv file, excluding the first row which should be column labels. It will only write out new data to the new csv file, it will not change the original csv file. For example:
```
python shuffle_csv.py data/evenly_distributed_reviews.csv data/filtered_reviews.csv
```

## Splitting Data into Train, Validation, and Test Sets
Once you have a filtered_reviews.csv file, you can call
```
python csv_splitter.py path_to_data_directory csv_file_name_no_suffix num_lines_to_split --percent_train=percent_train --percent_val=precent_val --percent_test=percent_test
```
For example, if there is a file called filtered_reviews.csv in a data/ directory and you want to use 1000 lines split 50%-25%-25% for train-validation-test, you can call
```
python csv_splitter.py data/ filtered_reviews 1000 --percent_train=0.5 --percent_val=0.25 --percent_test=0.25
```
This will create 3 new files, filtered_reviews_train.csv, filtered_reviews_val.csv, and filtered_reviews_test.csv in the data/ directory. Each of these new file will have the header from the input file and the relevant number of lines. So filtered_reviews_train.csv will have 50 lines, etc. The default split if you don't give those arguments is 60-20-20. You must put in all parameters if you are not using the default, and there is no checking to see if it adds up to 100%.

## Serializing the Data
If you want to package the train, validation, and test sets into a pickle file, call
```
python make_pickle_file.csv path_to_training_data path_to_validation_data path_to_testing_data path_to_pickle_file
```
For example:
```
python make_pickle_file.csv data/filtered_reviews_train.csv data/filtered_reviews_val.csv data/filtered_reviews_test.csv data/filtered_reviews.pickle
```
Currently none of our classifiers take in pickle files.

## Featurizing Data
When you have your filtered training, validation, and test csv files, you can featurize the inputs and generate both binary and multiclass labels for the data as follows:
```
from preprocess import *

# training data
preprocessor_train = Preprocessor('../data/filtered_reviews_train.csv')
preprocessor_train.cleanup()
dict = preprocessor_train.get_dictionary()
X_train, Y_train_multi, Y_train_binary = preprocessor_train.featurize(dict)

# test data
preprocessor_test = Preprocessor('../data/filtered_reviews_test.csv')
preprocessor_test.cleanup()
X_test, Y_test_multi, Y_test_binary = preprocessor_test.featurize(dict)
```
