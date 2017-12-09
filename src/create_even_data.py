import csv
import argparse

def get_even_data(input_csv_file, output_csv_file, num_classes, num_each_class, delimiter='\t'):
    if (not num_classes == 2) and (not num_classes == 5):
        print("Number of classes must be 2 or 5. Not processing data.")
        return

    num_binary_classes = 2
    positive_threshold = 3
    positive_review_index = 0
    negative_review_index = 1

    num_multi_classes = 5

    output_file = open(output_csv_file, 'wb+')
    output_writer = csv.writer(output_file, delimiter=delimiter)

    review_rating_index = 5

    class_counts = [0] * num_classes


    with open(input_csv_file, 'r') as fin:
        # Read in the column names from the input file
        col_names = fin.readline().strip().split(delimiter)
        output_writer.writerow(col_names)

        have_even_data = False
        for line in fin:
            row = line.strip().split(delimiter)
            rating = int(row[review_rating_index])

            if num_classes == num_binary_classes:
                if rating >= positive_threshold and class_counts[positive_review_index] < num_each_class:
                    class_counts[positive_review_index] += 1
                    output_writer.writerow(row)
                elif rating < positive_threshold and class_counts[negative_review_index] < num_each_class:
                    class_counts[negative_review_index] += 1
                    output_writer.writerow(row)
            elif num_classes == num_multi_classes:
                if class_counts[rating-1] < num_each_class:
                    class_counts[rating-1] += 1
                    output_writer.writerow(row)

            for class_count in class_counts:
                if class_count < num_each_class:
                    have_even_data = False
                    break
                have_even_data = True
            if have_even_data:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Split data into train, validation, and test sets.',
            )
    parser.add_argument(
            'input_csv_file_path',
            type=str,
            help='The name csv file to read in.',
            )
    parser.add_argument(
            'output_csv_file_path',
            type=str,
            help='The name csv file to write to.',
            )
    parser.add_argument(
            'total_num_classes',
            type=int,
            help='Whether you want to make binary or multiclass data. Must be 2 or 5',
            )
    parser.add_argument(
            'num_each_class',
            type=int,
            help='The number of reviews of each class.',
            )

    args = parser.parse_args()
    input_csv_file = args.input_csv_file_path
    output_csv_file = args.output_csv_file_path
    num_classes = args.total_num_classes
    num_each_class = args.num_each_class

    get_even_data(input_csv_file, output_csv_file, num_classes, num_each_class)