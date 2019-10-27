import pandas as pd

directory = "../../data/raw/plasmid_data/"
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
res.to_csv("../../data/processed/plasmid.tsv", sep="\t", index=False)
