# -*- coding: utf-8 -*-
"""Convert the Yelp Dataset Challenge dataset from json format to csv.

For more information on the Yelp Dataset Challenge please visit http://yelp.com/dataset_challenge

"""
import argparse
import collections
import csv
import json
import re
import pdb

def read_and_write_file(json_file_path, csv_file_path, column_names, delimiter=None):
    """Read in the json dataset file and write it out to a csv file, given the column names."""
    with open(csv_file_path, 'wb+') as fout:
        csv_file = csv.writer(fout, delimiter=delimiter)
        csv_file.writerow(list(column_names))
        with open(json_file_path) as fin:
            for line in fin:
                line_contents = json.loads(line)
                for k, v in line_contents.iteritems():
                    if isinstance(v, basestring):
                        line_contents[k] = re.sub(r'\r|\n', '', v)
                csv_file.writerow(get_row(line_contents, column_names))

def get_superset_of_column_names_from_file(json_file_path):
    """Read in the json dataset file and return the superset of column names."""
    column_names = set()
    with open(json_file_path) as fin:
        for line in fin:
            line_contents = json.loads(line)
            column_names.update(
                    set(get_column_names(line_contents).keys())
                    )
    return column_names

def get_column_names(line_contents, parent_key=''):
    """Return a list of flattened key names given a dict.

    Example:

        line_contents = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }

        will return: ['a.b', 'a.c']

    These will be the column names for the eventual csv file.

    """
    column_names = []
    for k, v in line_contents.iteritems():
        column_name = "{0}.{1}".format(parent_key, k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            column_names.extend(
                    get_column_names(v, column_name).items()
                    )
        else:
            column_names.append((column_name, v))
    return dict(column_names)

def get_nested_value(d, key):
    """Return a dictionary item given a dictionary `d` and a flattened key from `get_column_names`.
    
    Example:

        d = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }
        key = 'a.b'

        will return: 2
    
    """
    if '.' not in key:
        if key not in d:
            return None
        return d[key]
    base_key, sub_key = key.split('.', 1)
    if base_key not in d:
        return None
    sub_dict = d[base_key]
    return get_nested_value(sub_dict, sub_key)

def get_row(line_contents, column_names):
    """Return a csv compatible row given column names and a dict."""
    row = []
    for column_name in column_names:
        line_value = get_nested_value(
                        line_contents,
                        column_name,
                        )
        if isinstance(line_value, unicode):
            row.append('{0}'.format(line_value.encode('utf-8')))
        elif line_value is not None:
            row.append('{0}'.format(line_value))
        else:
            row.append('')
    return row

def write_txt_file(txt_file_path, data_name, data_to_write, delimiter=None):
    with open(txt_file_path, 'wb+') as fout:
        fout.write(data_name + "\n")
        for data_point in data_to_write:
            fout.write(data_point + "\n")

'''
Creates a set with values from the column with target_col_name if it contains the right filtering values and/or categories

json_file_path is a string of the json file to parse.
target_col_name is a string of a single column name.
scalar_filter_col_names is a list of strings of single-valued column names to filter by,
    for example ["state", "name"].
scalar_filter_values is a list of values to filter for. It should be the same length as scalar_filter_col_names,
    and each value should correspond to the same index of column name, including its datatype
filter_categories is a list of strings to filter by in the "categories" field of each line. If the json file doesn't
    have a "categories" field, then pass in an empty list


Returns a set of the values in the target_col_name in the json_file that match the given criteria
'''
def create_filtered_set(json_file_path, target_col_name, scalar_filter_col_names, scalar_filter_values, filter_categories):
    data = set()
    with open(json_file_path) as fin:
        for line in fin:
            line_contents = json.loads(line)
            record_data = True
            for count in range(len(scalar_filter_col_names)):
                col_name = scalar_filter_col_names[count]
                if line_contents[col_name] != scalar_filter_values[count]:
                    record_data = False
            for category in filter_categories:
                if category not in line_contents["categories"]:
                    record_data = False
            if record_data:
                data.add(line_contents[target_col_name])
    return data

if __name__ == '__main__':
    """Convert a yelp dataset file from json to csv."""

    parser = argparse.ArgumentParser(
            description='Convert Yelp Dataset Challenge data from JSON format to CSV.',
            )

    parser.add_argument(
            'json_file',
            type=str,
            help='The json file to convert.',
            )

    parser.add_argument(
            'csv_file',
            type=str,
            help='The csv file to convert to.',
            )

    parser.add_argument(
            'filtered_txt_file',
            type=str,
            help='The text file to write out filtered data',
            )

    args = parser.parse_args()
    json_file = args.json_file

    ## Create new unfiltered csv file ##
    csv_file = args.csv_file

    print "getting column names"
    column_names = get_superset_of_column_names_from_file(json_file)

    print "reading and writing file"
    read_and_write_file(json_file, csv_file, column_names, delimiter='\t')

    ## Create filtered business id set and write it to a file ##
    filtered_txt_file = args.filtered_txt_file

    target_column_name = "business_id"
    scalar_filter_columns = ["state"]
    scalar_filter_values = ["AZ"]
    filter_categories = ["Restaurants"]

    print "filtering businesses"
    filtered_business_ids = create_filtered_set(json_file, target_column_name, scalar_filter_columns, scalar_filter_values, filter_categories)
    print "number of filtered businesses:", len(filtered_business_ids)

    print "writing filtered business ids to text file"
    write_txt_file(filtered_txt_file, "business_id", filtered_business_ids)
