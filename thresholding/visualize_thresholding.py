#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-f", "--file", help="The file containing genome data.")
args = parser.parse_args()


f = "thresholding.csv" if not args.file else args.file
t = pd.read_csv(f).sort_values(by=["Threshold"])
t.to_csv(f, index=False)

t.plot(x="Threshold", y=["Total", "True Positives", "False Positives"], kind="line")

plt.show()
