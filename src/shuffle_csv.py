import random
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Randomly shuffle a csv file except the first row.',
            )

    parser.add_argument(
            'csv_in',
            type=str,
            help='The csv file to shuffle.',
            )

    parser.add_argument(
            'csv_out',
            type=str,
            help='The new shuffled csv file.',
            )

    args = parser.parse_args()

    fin = open(args.csv_in, "r")
    li = fin.readlines()
    fin.close()
    names = li[0]
    li = li[1:]
    
    random.shuffle(li)

    fout = open(args.csv_out, "w")
    fout.write(names)
    fout.writelines(li)
    fout.close()