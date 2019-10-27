#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-f", "--file", help="The file containing thresholding data.")
args = parser.parse_args()


f = "thresholding.csv" if not args.file else args.file
t = pd.read_csv(f).sort_values(by=["Threshold"])
t.to_csv(f, index=False)

# t.plot(x="Threshold", y=["Total", "True Positives", "False Positives"], kind="line")
# plt.title("Feature Thresholding")

t["tp/fp"] = t["True Positives"] / t["False Positives"]
print(t["Threshold"].unique())
print(t.values)
plt.plot(x=t["Threshold"].unique(), y=t.values)
print(t)
t.plot(x="Threshold", y="tp/fp")
plt.title("Feature tp/fp Ratio")
plt.show()
