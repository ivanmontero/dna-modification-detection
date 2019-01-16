# Filters out noise

import pandas as pd
import numpy as np 
import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-f", "--file", help="The file containing genome data.")
args = parser.parse_args()
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# plt.ioff()

# The chromosome file to parse
FILE = "l_tarentolae.tsv"
if args.file:
    FILE = args.file

# Create the pandas file
TABLE = pd.read_table(FILE).dropna()

FOLD_THRESHOLD = 10

def find_false_positives(threshold):
    false_positives = 0
    for index, row in TABLE.iterrows():
        if row["IPD Top Ratio"] >= ipd_threshold and row["Fold Change"] >= FOLD_THRESHOLD:
            false_positives += 1
    return (threshold, false_positives)

from multiprocessing import Pool
if __name__ == "__main__":
    params = []
    ipd_threshold = 0.0
    while ipd_threshold < TABLE["IPD Top Ratio"].max():
        params.append(ipd_threshold)
        ipd_threshold += 0.1

    pool = Pool(os.cpu_count())
    data = pool.map(find_false_positives, params)

    output_file = open("threshold_fp_rate.csv", "w+")
    for datum in data:
        output_file.write(datum[0] + "," + datum[1] + "\n")
    output_file.close()


