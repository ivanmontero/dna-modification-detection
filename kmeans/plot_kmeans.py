import pandas as pd
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
# plt.ioff()

DIR = "trunc_r/"
FILE = "kmeans_trunc_r.csv"

# Accuracy is defined as follows:
#   acc = # correct classifications / total
# A "classification" is determined by the greatest label in one
# cluster. A "correct classification" is a single point in a
# cluster whose classification matches the label of the point.
# 
# param: Table containing a single kmeans run
def acc(tab, n_clusters):
    # print(tab)
    n_correct = 0
    n_total = tab["total"].sum()
    for i in range(tab.shape[0]):
        row = tab.iloc[i]
        classification = "tp" if (row["tp"] / row["total"])>.5 else "fp"
        # print(classification + " " + str(row["tp"] / row["total"]))
        n_correct += row[classification]
    print(str(n_correct) + " " + str(n_total))
    return n_correct / n_total

# Distribution is defined as follows:
#   dist = (\sum_cluster |#points in cluster - avg|) / n_clusters
# where
#   avg = total points / n_clusters
# 
# param: Table containing a single kmeans run
def dist(tab, n_clusters):
    sum_dist = 0
    n_total = tab["total"].sum()
    avg = n_total / n_clusters
    for i in range(tab.shape[0]):
        row = tab.iloc[i]
        sum_dist += abs(row["total"]-avg)
    return sum_dist / n_clusters


kmeans = pd.read_csv(FILE)

acc_map = {}
for radius in kmeans["radius"].unique():
    acc_map[radius] = {}
    for n_clusters in kmeans["n_clusters"].unique():
        acc_map[radius][n_clusters] = {}
        curr = kmeans[(kmeans["radius"] == radius) &
                      (kmeans["n_clusters"] == n_clusters)]
        # Do computation here
        acc_map[radius][n_clusters]["acc"] = acc(curr, n_clusters)
        acc_map[radius][n_clusters]["dist"] = dist(curr, n_clusters)

print(acc_map)

for radius in kmeans["radius"].unique():
    curr = kmeans[(kmeans["radius"] == radius)]
    plt.plot(curr["n_clusters"], [acc_map[radius][i]["acc"] for i in curr["n_clusters"]])
    plt.xlabel("# Of Clusters")
    plt.ylabel("Accuracy")
    plt.title("Sequence Radius " + str(radius))
    plt.savefig(DIR + "r%d_acc.png" % (radius,), dpi=400)
    plt.show()

# for radius in kmeans["radius"].unique():
#     curr = kmeans[(kmeans["radius"] == radius)]
#     plt.plot(curr["n_clusters"], [acc_map[radius][i]["dist"] for i in curr["n_clusters"]])
#     plt.xlabel("# Of Clusters")
#     plt.ylabel("Distribution")
#     plt.title("Sequence Radius " + str(radius))
#     plt.show()

maxima = [[5, 8], [10, 9], [15, 9], [20, 9], [25, 9], [50,9], [100, 12]]

for m in maxima:
    p = kmeans[(kmeans["radius"] == m[0]) & (kmeans["n_clusters"] == m[1])]

    plt.subplot(2, 1, 1)
    plt.bar(p["cluster_id"], p["tp"] / p["total"])
    plt.title("Sequence Radius " + str(m[0]) + ", " + str(m[1]) + " Clusters")

    plt.subplot(2, 1, 2)

    plt.bar(p["cluster_id"], p["total"])

    # plt.savefig("r%d_c%d_kmeans.png" % (m[0], m[1]), dpi=400)
    plt.show()
    plt.cla()
    plt.close("all")


# plt.bar(, [])
# plt.ylabel("Average Positives")
# plt.title("K-Means")

# plt.subplot(2, 1, 2)
# plt.bar([x for x in range(n_clusters)], c_size)
# plt.ylabel("# of Peaks")

# plt.xlabel("Cluster #")
    
# # plt.scatter(dim_red[:,0], dim_red[:,1], c=colors, s=1)
    
# # Save the plot
# plt.savefig(PLOT_DIR + name + ".png")
# plt.cla()
# plt.close("all")