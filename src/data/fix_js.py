import pandas as pd
from multiprocessing import Pool, Lock
import numpy as np 
import os
import math


mapping = {
    "25L_PLASMID_corrected": "cSSR 25.2L WT Lt",
    "cSSR_12_pGEM_a-neo-a cSSR 12 insert (bases 2 to 411) inserted into 10x GGGTTA pGEM a-neo-a (bases 2690 to 2627)" : "cSSR 12.1 WT Lt",
    "TTA pGEM plasmid 5581 bp" : "GGGTTAX10 WT Lt",
    "25S_PLASMID" : "cSSR 25.2S WT Lt"
}

mapping_strand = {
    "top" : 0,
    "bottom" : 1
}

p = pd.read_csv("../data/plasmid.tsv", sep="\t")
# print(p["refName"].unique())
js = pd.read_csv("../data/Js_filtered.csv")
print(p)
# Account for offset in the Js file
js["Position"] = js["Position"].apply(lambda x: x - 1)

p["refName"] = p["refName"].map(mapping)
js["Strand"] = js["Strand"].map(mapping_strand)
print(p["refName"].unique())
print(p)
p = p.dropna()
print(p["refName"].unique())
p["J"] = 0
for index, row in js.iterrows():
    val = p[(p["refName"] == row["Plasmid"]) & (p["tpl"] == row["Position"]) & (p["strand"] == row["Strand"])]
    if val.shape[0] != 1:
        print(f"No row found for: {row}")
        continue
    # val.iloc[0]["J"] = 1
    p.at[val.index, "J"] = 1

print(p)
print(p["J"].sum())
print(js.shape)
p = p.rename(columns={"refName": "plasmid", "tpl": "position"})
p.to_csv("../../data/processed/plasmid_and_j.csv", index=False)
