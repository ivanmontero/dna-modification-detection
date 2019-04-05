
import pandas as pd
import numpy as np 
from operator import add, truediv, sub
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-c", "--centers", help="The file containing center files.")
parser.add_argument("-b", "--bases", help="The file containing bases")
parser.add_argument("-o", "--outdir", help="The directory to hold output.")
args = parser.parse_args()

centers = pd.read_csv(args.centers)
bases = pd.read_csv(args.bases)
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
print(bases)

def write(c, name):
    to_drop = centers.isin(c).iloc[:,0]
    s = bases[to_drop].values

    f = open(args.outdir + name + ".txt", "w+")
    for row in s:
        for b in row:
            f.write(b)
        f.write('\n')
    f.close()

tp_top = centers[(centers["Fold Change"] > 10) &
                 (centers["IPD Top Ratio"] > centers["IPD Bottom Ratio"])]
write(tp_top, "tp_top_sequence")

tp_bottom = centers[(centers["Fold Change"] > 10) &
                    (centers["IPD Top Ratio"] < centers["IPD Bottom Ratio"])]
write(tp_bottom, "tp_bottom_sequence")

fp_top = centers[(centers["Fold Change"] <= 10) &
                 (centers["IPD Top Ratio"] > centers["IPD Bottom Ratio"])]
write(fp_top, "fp_top_sequence")

fp_bottom = centers[(centers["Fold Change"] <= 10) &
                    (centers["IPD Top Ratio"] < centers["IPD Bottom Ratio"])]
write(fp_bottom, "fp_bottom_sequence")
