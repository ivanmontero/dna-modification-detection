#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-f", "--file", help="The file containing genome data.")
args = parser.parse_args()


f = "thresholding_filter_fold.csv" if not args.file else args.file
t = pd.read_csv(f).sort_values(by=["threshold"])
t = t[t["threshold"] < 5]

t.plot(x="threshold", y="positives", kind="line")

plt.show()