import csv
import argparse

def split(data_path, csv_file_name, total_num_lines, percent_train, percent_val, percent_test, delimiter='\t'):
    input_file_name = data_path + csv_file_name + '.csv'
    train_file_name = data_path + csv_file_name + '_train.csv'
    val_file_name = data_path + csv_file_name + '_val.csv'
    test_file_name = data_path + csv_file_name + '_test.csv'

    # Calculate the number of lines to write to each file, rounding down to the
    # nearest integer
    train_lines = int(percent_train * total_num_lines)
    val_lines = int(percent_val * total_num_lines)
    test_lines = int(percent_test * total_num_lines)

    print train_lines, val_lines, test_lines

    with open(input_file_name, 'r') as fin:
        # Open the new files - remember to close them at the end
        train_file = open(train_file_name, 'wb+')
        val_file = open(val_file_name, 'wb+')
        test_file = open(test_file_name, 'wb+')

        train_writer = csv.writer(train_file, delimiter=delimiter)
        val_writer = csv.writer(val_file, delimiter=delimiter)
        test_writer = csv.writer(test_file, delimiter=delimiter)

        # Read in the column names from the input file
        col_names = fin.readline().strip().split(delimiter)

        # Write the column names as a header to each of the new files
        train_writer.writerow(col_names)
        val_writer.writerow(col_names)
        test_writer.writerow(col_names)

        # Write each line in the input file to the relevant new file
        current_line = 1
        for line in fin:
            row = line.strip().split(delimiter)
            if current_line <= train_lines:
                train_writer.writerow(row)
            elif current_line <= train_lines + val_lines:
                val_writer.writerow(row)
            elif current_line <= train_lines + val_lines + test_lines:
                test_writer.writerow(row)
            else:
                break
            current_line += 1

        # Close the new files
        train_file.close()
        val_file.close()
        test_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Split data into train, validation, and test sets.',
            )
    parser.add_argument(
            'data_path',
            type=str,
            help='Directory of the data file',
            )
    parser.add_argument(
            'csv_file_name',
            type=str,
            help='The name csv file to split WITHOUT the .csv suffix.',
            )
    parser.add_argument(
            'total_num_lines',
            type=int,
            help='The total number of lines in the input file to split up',
            )
    parser.add_argument(
            '--percent_train',
            type=float,
            help='The percentage of the lines to split into training data as a decimal, \
                    so 50%% should be entered as 0.5',
            )
    parser.add_argument(
            '--percent_val',
            type=float,
            help='The percentage of the lines to split into validation data as a decimal, \
                    so 50%% should be entered as 0.5',
            )
    parser.add_argument(
            '--percent_test',
            type=float,
            help='The percentage of the lines to split into test data as a decimal, \
                    so 50%% should be entered as 0.5',
            )

    args = parser.parse_args()
    data_path = args.data_path
    csv_file_name = args.csv_file_name
    total_num_lines = args.total_num_lines

    percent_train = 0.6
    percent_val = 0.2
    percent_test = 0.2

    if args.percent_train:
        percent_train = args.percent_train
    if args.percent_val:
        percent_val = args.percent_val
    if args.percent_test:
        percent_test = args.percent_test

    split(data_path, csv_file_name, total_num_lines, percent_train, percent_val, percent_test)