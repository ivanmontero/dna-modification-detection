import pandas as pd

from argparse import ArgumentParser

directory = "../data/original/plasmid_data/"
# 1
# 3
# 7
# 8
res = None
for i in [1, 3, 7, 8]:
    try:
        table = pd.read_csv(f"{directory}{i}.csv")
    except:
        continue
    t = table[["refName", "tpl", "strand", "base", "ipdRatio"]]
    if res is None:
        res = t
    else:
        res = pd.concat([res, t])
res.to_csv("plasmid.tsv", sep="\t", index=False)
