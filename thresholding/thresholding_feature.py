# Output values to stdout

import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, Lock
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-f", "--file", help="The file containing genome data.")
parser.add_argument("-o", "--outfile", help="The file to output the new table.")
parser.add_argument("-g", "--generate", help="Generate data.", action="store_true")
args = parser.parse_args()


def dist(features, begin, end, step):
    print("%f-%f" % (begin, end))
    t = begin
    data = ""
    while t < end:
        res = features[features["Feature"] > t]
        res_total = res.shape[0]
        res_tp = res["True Positive"].sum()
        res_fp = res_total - res_tp
        data += "%f,%d,%d,%d\n" % (t, res_total, res_tp, res_fp)
        t += step
    lock.acquire()
    f = open(args.outfile, "a")
    f.write(data)
    f.close()
    lock.release()


if args.generate:
    # The chromosome file to parse
    FILE = "l_tarentolae.tsv"
    if args.file:
        FILE = args.file

    # Create the pandas file
    FOLD_THRESHOLD = 10
    TABLE = pd.read_table(FILE).fillna(0)

    f = open(args.outfile, "w+")
    f.write("%s,%s,%s,%s\n" % ("Chromosome", "Position", "Feature", "True Positive"))
    for c in TABLE["Chromosome"].unique():
        tab = TABLE[TABLE["Chromosome"] == c]
        fold_sequence = tab["Fold Change"].values
        ipd_sequence = tab["IPD Top Ratio"].values
        for i in range(5, len(ipd_sequence)):
            f.write("%s,%d,%f,%d\n" % (c,
                                     i,
                                     ipd_sequence[i-6] + ipd_sequence[i-2] + ipd_sequence[i],
                                     1 if fold_sequence[i] > FOLD_THRESHOLD else 0))
    f.close()

def wrapper(args):
    dist(*args)

def init(l):
    global lock
    lock = l

if __name__ == '__main__' and not args.generate:
    FILE = "features.tsv"
    if args.file:
        FILE = args.file

    f = open(args.outfile, "w+")
    f.write("Threshold,Total,True Positives,False Positives\n")
    f.close()

    step = .01
    features = pd.read_csv(FILE)
    print(features.describe())

    params = []
    for i in range(180):
        params.append((features,float(i),float(i+1),0.01))

    l = Lock()

    pool = Pool(os.cpu_count(), initializer=init, initargs=(l,))
    pool.map(wrapper, params)

    # Sort
    print("sorting")
    result = pd.read_csv(args.outfile)
    result = result.sort_values(by=["Threshold"])
    result.to_csv(args.outfile, index=False)