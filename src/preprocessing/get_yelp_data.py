from numpy import genfromtxt
import numpy as np
import argparse
import pdb
import csv
import sys
import re
from yelp_data_constants import *


''' Helper functions '''
def construct_filtered_set(txt_file_path):
    title = ""
    filtered_values = set()
    first_line = True
    with open(txt_file_path, 'r') as fin:
        for line in fin:
            if first_line:
                title = line.strip()
                first_line = False
            else:
                filtered_values.add(line.strip())
    return title, filtered_values

def write_data_to_csv_file(input_data, csv_file_path, names, delimiter='\t'):
    with open(csv_file_path, 'w+') as fout:
        csv_file = csv.writer(fout, delimiter=delimiter)
        csv_file.writerow(names)
        for review in input_data:
            csv_file.writerow(review)

def get_filtered_data(data, filter_column_index, filtered_value_set):
    filtered_data = []
    for d in data:
        column_decoded = strip_bytes(d[filter_column_index])
        if column_decoded in filtered_value_set:
            filtered_data.append(d)
    return filtered_data

def get_data(csv_file, names):
    print("getting data")
    filling = None
    data = genfromtxt(csv_file, dtype=None, names=names, \
                             delimiter='\t', skip_header=1, comments=None, \
                             filling_values=filling)
    print("done getting data")
    return data

def get_data_big(input_csv_file, names, filtered_column_name, \
                     filtered_value_set, output_csv_file):
    with open(input_csv_file, 'r') as fin:
        print("reading from big csv")
        csv.field_size_limit(sys.maxsize)
        reader = csv.DictReader(fin, delimiter='\t') #, fieldnames=names)

        print("writing csv")
        with open(output_csv_file, 'w') as fout:
            writer = csv.writer(fout, delimiter='\t')
            writer.writerow(reader.fieldnames)
            for row in reader:
                column_decoded = strip_bytes(row[filtered_column_name])
                if column_decoded in filtered_value_set:
                    writer.writerow(row.values())
                    # writer.writerow(strip_bytes(v) for v in row.values())
    print("done writing to csv")

def strip_bytes(s):
    if 'b\'' in str(s):
        r = re.findall(r"b'(.*?)'", str(s))
        if len(r) > 0:
            return r[0]
    return s


''' Review data '''
def get_review_data(csv_file):
    return get_data(csv_file, REVIEW_NAMES)

def get_filtered_review_data(review_data, filtered_value_set):
    return get_filtered_data(review_data, REVIEW_BUSID_IDX, filtered_value_set)


''' Business data '''
def get_business_data(csv_file):
    return get_data(csv_file, BUSINESS_NAMES)

def get_filtered_business_data(business_data, filtered_value_set):
    return get_filtered_data(business_data, BUSINESS_BUSID_IDX, filtered_value_set)


''' User data '''
def get_user_data(csv_file):
    return get_data(csv_file, USER_NAMES)

def get_filtered_user_data(user_data, filtered_value_set):
    return get_filtered_data(user_data, USER_USERID_IDX, filtered_value_set)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Filter Yelp data based on an attribute and create filtered csv file.',
            )
    parser.add_argument(
            'file_type',
            type=str,
            help='Review or business or user',
            )
    parser.add_argument(
            'csv_file',
            type=str,
            help='The csv file to load.',
            )
    parser.add_argument(
            'filter_values',
            type=str,
            help='The text files of values to filter reviews.',
            )
    parser.add_argument(
            'filtered_file_name',
            type=str,
            help='Filepath for csv file containing filtered reviews.',
            )

    args = parser.parse_args()
    file_type = args.file_type
    csv_file = args.csv_file
    filtered_file_name = args.filtered_file_name
    filter_values = args.filter_values

    if file_type == 'review':
        print("review type")
        data = get_review_data(csv_file)
        if args.filtered_file_name:
            print("constructing filtered set")
            label, business_ids = construct_filtered_set(filter_values)
            if label == "business_id":
                print("getting filtered review data")
                filtered_data = get_filtered_review_data(data, business_ids)
                print("writing filtered review data to csv file")
                write_data_to_csv_file(filtered_data, filtered_file_name, REVIEW_NAMES)

    elif file_type == 'business':
        print("business type")
        data = get_business_data(csv_file)
        print("constructing filtered set")
        label, business_ids = construct_filtered_set(filter_values)
        filtered_data = get_filtered_business_data(data, business_ids)
        print("writing filtered business data to csv file")
        write_data_to_csv_file(filtered_data, filtered_file_name, BUSINESS_NAMES)
        print("wrote filtered business data to ", filtered_file_name)

    elif file_type == 'user':
        print("constructing filtered set")
        label, user_ids = construct_filtered_set(filter_values)
        get_data_big(csv_file, USER_NAMES, label, \
                     user_ids, filtered_file_name)

        '''
        print("user type")
        data = get_user_data(csv_file)
        print("constructing filtered set")
        label, user_ids = construct_filtered_set(filter_values)
        filtered_data = get_filtered_user_data(data, user_ids)
        print("writing filtered user data to csv file")
        write_data_to_csv_file(filtered_data, filtered_file_name, USER_NAMES)
        print("wrote filtered user data to ", filtered_file_name)
        '''
