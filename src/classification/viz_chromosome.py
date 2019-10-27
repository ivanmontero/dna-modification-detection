import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import Index

g = pd.read_csv("../data/l_tarentolae.tsv", sep="\t")
c1 = g[g["Chromosome"] == "LtaP_01"]
j = pd.read_csv("j_positions.tsv", sep="\t")
j1 = j[j["Chromosome"] == "LtaP_01"]
new_index = Index(np.arange(j1["Position"].max()), name="Position")
j1 = j1.set_index("Position")
j1 = j1.reindex(new_index)
j1 = j1.reset_index()
j1 = j1.fillna(0)

plt.plot(c1["Position"], c1["Top IPD Ratio"])
plt.plot(c1["Position"], c1["Fold Change"])
plt.plot(j1["Position"], j1["Delta"])

# o.sort_values(by=["refName", "tpl"], inplace=True)
# plt.hist(o[o["J_delta"] != 0]["J_delta"], bins=30)
# plt.title("Plasmid Drop in Probability Distribution")
# plt.show()

# l = pd.read_csv("j_positions.tsv", sep="\t")
# plt.hist(l[l["Delta"] <= 0]["Delta"], bins=30)
# plt.title("Genome Drop in Probability Distribution")
# plt.show()

for n in o["refName"].unique():
    p = o[o["refName"] == n]

    plt.plot(p["tpl"], p["ipdRatio"], label="IPD Ratio")
    plt.plot(p["tpl"], p["J_pred"], label="J Probability")
    plt.plot(p["tpl"], p["J_delta"], label="Probability Drop")
    plt.legend()
    plt.title(n)
    plt.show()