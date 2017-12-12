import argparse
import collections
import csv
import json
import re
import pdb
from get_yelp_data import get_data
from yelp_data_constants import *


'''
Creates a set with values from the column with output_col_name if it contains the right 
filtering values and/or categories

json_file_path is a string of the json file to parse.
output_col_name is a string of a single column name.
scalar_filter_col_names is a list of strings of single-valued column names to filter by, for example ["state", "name"].
scalar_filter_values is a list of values to filter for. It should be the same length as 
scalar_filter_col_names, and each value should correspond to the same index of column name, including its data type
filter_categories is a list of strings to filter by in the "categories" field of each line. 
If the json file doesn't have a "categories" field, then pass in an empty list

Returns a set of the values in the target_col_name in the json_file that match the given criteria
'''
def create_filtered_set_json(json_file_path, scalar_filter_col_names, \
				scalar_filter_values, filter_categories, output_col_name):
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
                data.add(line_contents[output_col_name])
    return data

def create_filtered_set_csv(csv_file_path, scalar_filter_col_names, \
                                scalar_filter_values, filter_categories, \
				output_col_name, names):
	data = get_data(csv_file_path, names)
	filtered_data = set()
	for d in data:
		record_data = True
		for count in range(len(scalar_filter_col_names)):
			col_name = scalar_filter_col_names[count]
			if d[col_name] != scalar_filter_values[count]:
				record_data = False
		for category in filter_categories:
			if category not in d["categories"]:
				record_data = False
		if record_data:
			filtered_data.add(d[output_col_name])
	return filtered_data

def write_txt_file(txt_file_path, data_name, data_to_write, delimiter=None):
	with open(txt_file_path, 'w+') as fout:
	    fout.write(data_name + "\n")
	    for data_point in data_to_write:
		    if isinstance(data_point, bytes):
			    data_point = data_point.decode()
		    fout.write(str(data_point) + "\n")

def get_filtered_id_set(json_file_path, filter_column_name, filtered_value_set, id_column_name):
    ids = set()
    with open(json_file_path) as fin: 
	    for line in fin:
		    d = json.loads(line)
		    if d[filter_column_name] in filtered_value_set:
			    id = d[id_column_name]
			    ids.add(id)
    return ids


if __name__ == '__main__':
    """Get filtered ids based on specified filter criterion."""

    parser = argparse.ArgumentParser(
            description='Create filter for Yelp data',
            )

    parser.add_argument(
            'input_file',
            type=str,
            help='The input csv or json file to filter from.',
            )

    parser.add_argument(
	    'file_type',
	    type=str,
	    help='File type of input file: review or business or user',
	    )

    parser.add_argument(
            'filtered_txt_file',
            type=str,
            help='The text file to write out filtered data',
            )

    parser.add_argument(
	    '--csv',
	    required=False,
	    default=False,
	    action='store_true',
	    help='input file is csv',
	    )

    # parse arguments
    args = parser.parse_args()
    input_file = args.input_file
    filtered_txt_file = args.filtered_txt_file
    use_csv = args.csv
    file_type = args.file_type

    # specify filters 
    scalar_filter_columns = []
    scalar_filter_values = []
    filter_categories = []
    output_column_name = "user_id"

    # get filtered ids
    print("filtering for: %s" % output_column_name)
    if use_csv:
	    filtered_ids = create_filtered_set_csv(input_file, \
			   scalar_filter_columns, \
			   scalar_filter_values, filter_categories, \
			   output_column_name, type_to_names[file_type])
    else:	    
	    filtered_ids = create_filtered_set_json(input_file, \
			   scalar_filter_columns, scalar_filter_values, 
                           filter_categories, output_column_name)

    # write ids to text file
    print("number of filtered %s:%d" % (output_column_name, len(filtered_ids)))
    print("writing filtered %s to text file" % output_column_name)
    write_txt_file(filtered_txt_file, output_column_name, filtered_ids)
    print("wrote filtered %s to %s" % (output_column_name, filtered_txt_file))
