import pandas as pd
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c","--chromosome")
parser.add_argument("-f","--file")
args = parser.parse_args()

table = pd.read_table(args.file)

# print(table)
print(table.describe())
# print(table["Chromosome"])
# print(table.dtypes)


c1 = table[table["Chromosome"] == args.chromosome]
print(c1.describe())

c1.plot(x="Position", y=["Fold Change", "IPD Top Ratio", "IPD Bottom Ratio"])
# upper_bound =  c1["Fold Change"].mean()
# plt.plot([0, c1.shape[0]], [upper_bound, upper_bound])
# plt.show()
plt.savefig("test.png")
