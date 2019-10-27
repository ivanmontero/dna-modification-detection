# TODO: Refactor to use sequences

import pandas as pd
import numpy as np 
import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-f", "--file", help="The file containing genome data.")
args = parser.parse_args()

# The chromosome file to parse
FILE = "l_tarentolae.tsv"
if args.file:
    FILE = args.file

# Create the pandas file
FOLD_THRESHOLD = 10
TABLE = pd.read_table(FILE).dropna()
TABLE = TABLE[TABLE["Fold Change"] > FOLD_THRESHOLD]


def find_false_positives(threshold):
    positives = TABLE[TABLE["IPD Top Ratio"] > threshold].shape[0]
    # Gives number of rows that are alse positives.
    # false_positives = postives[TABLE["Fold Change"] < FOLD_THRESHOLD].shape[0]

    print(str(threshold) + "," + str(positives))

from multiprocessing import Pool
if __name__ == "__main__":
    print("threshold,positives")
    params = []
    ipd_threshold = 0.0
    while ipd_threshold < 20:
        params.append(ipd_threshold)
        ipd_threshold += 0.1

    pool = Pool(os.cpu_count())
    pool.map(find_false_positives, params)

