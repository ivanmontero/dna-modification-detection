import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

o = pd.read_csv("plasmid_w_predictions.tsv", sep="\t")
# o.sort_values(by=["refName", "tpl"], inplace=True)
# plt.hist(o[o["J_delta"] != 0]["J_delta"], bins=30)
# plt.title("Plasmid Drop in Probability Distribution")
# plt.show()

# l = pd.read_csv("j_positions.tsv", sep="\t")
# plt.hist(l[l["Delta"] <= 0]["Delta"], bins=30)
# plt.title("Genome Drop in Probability Distribution")
# plt.show()

# pred to plas_and_j
mapping = {
    "25L_PLASMID_corrected": "cSSR 25.2L WT Lt",
    "cSSR_12_pGEM_a-neo-a cSSR 12 insert (bases 2 to 411) inserted into 10x GGGTTA pGEM a-neo-a (bases 2690 to 2627)" : "cSSR 12.1 WT Lt",
    "TTA pGEM plasmid 5581 bp" : "GGGTTAX10 WT Lt",
    "25S_PLASMID" : "cSSR 25.2S WT Lt"
}
plt.rcParams["figure.figsize"] = (12, 7)

for n in o["refName"].unique():
    p = o[o["refName"] == n]
    # p = p[p["strand"] == 0]
    if n not in mapping.keys():
        continue

    ax1 = plt.subplot(1,1,1)

    ax1.plot(p["tpl"], p["ipdRatio"], label="IPD Ratio")
    ax1.plot(p["tpl"], p["J_pred"], label="J Probability")
    ax1.plot(p["tpl"], p["J_delta"], label="Probability Drop")
    ax1.set_xlabel("Position")
    # ax1.set_xticks(p["tpl"])

    ax2 = ax1.twiny()
    ax2.set_xticks(p[p["base"] == "T"]["tpl"], "T")
    ax2.set_xlabel("T")
    ax2.set_xlim(ax1.get_xlim())
    # ax2.xaxis.set_ticks_position("bottom")
    # ax2.xaxis.set_label_position("bottom")
    ax2.tick_params(
        axis='x',          # changes apply to the x-axis
        which='major',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labeltop=False) # labels along the bottom edge are off
    # ax2.spines['bottom'].set_position(('outward', 36))
    p_j = pd.read_csv("../data/plasmid_and_j.csv")
    js = p_j[(p_j["plasmid"] == mapping[n]) & (p_j["J"] == 1) & (p_j["strand"] == 0)] # 0
    print(js)
    for i in js["position"]:
        ax1.axvline(i, color="black")


    ax1.legend()
    plt.title(n)
    plt.show()