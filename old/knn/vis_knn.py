import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-f", "--file", help="The file containing knn data.")
args = parser.parse_args()

t = pd.read_csv(args.file).sort_values(by=["radius", "k"])
# t.plot(x="k", y="acc", kind="line")
for r in t["radius"].unique():
    rt = t[t["radius"] == r].sort_values(by=["k"])
    # rt.plot(x="k", y="acc", kind="line", label="%d" % (r,))
    # for k in t["k"].unique():
    #     rt = t[t["radius"] == r]
    #     rt.plot(x=)
    print(rt["k"].values)
    print(rt["acc"].values)
    plt.plot(rt["k"].values, rt["acc"].values, label="r=%d" % (r,))
    # plt.show()
plt.title("K-Nearest Neighbors: Bottom Strand")
plt.legend()
plt.show()

# f = "thresholding.csv" if not args.file else args.file
# t = pd.read_csv(f).sort_values(by=["Threshold"])
# t.to_csv(f, index=False)

# # t.plot(x="Threshold", y=["Total", "True Positives", "False Positives"], kind="line")
# # plt.title("Feature Thresholding")

# t["tp/fp"] = t["True Positives"] / t["False Positives"]
# print(t["Threshold"].unique())
# print(t.values)
# plt.plot(x=t["Threshold"].unique(), y=t.values)
# print(t)
# t.plot(x="Threshold", y="tp/fp")
# plt.title("Feature tp/fp Ratio")
# plt.show()
