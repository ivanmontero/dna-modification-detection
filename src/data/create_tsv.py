import pandas as pd
from multiprocessing import Pool, Lock
import numpy as np 
import os

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-bases", "--bases", default="../../data/raw/LtaP_PB.genome.fasta")
parser.add_argument("-sizes", "--sizes", default="../../data/raw/LtaP_PB.sizes.txt")
parser.add_argument("-ipd", "--ipd", default="../../data/raw/ipd-h5.txt")
parser.add_argument("-fold", "--fold", default="../../data/raw/JM083.fold-change.txt")
parser.add_argument("-o", "--outfile", default="../../data/processed/l_tarentolae.tsv")
parser.add_argument("-parallel", "--parallel", action="store_true")
args = parser.parse_args()

f_bases = open(args.bases, "r")
# f_bases_sizes = open("../data/original/LtaP_PB.sizes.txt", "r")
bases = {}
c_chromosome = None
for line in f_bases:
    line = line.rstrip()
    if line[0] == '>':
        c_chromosome = line[1:]
        bases[c_chromosome] = []
    else:
        bases[c_chromosome].extend(list(line))
f_bases_sizes = open(args.sizes, "r")
sizes = {}
offsets = {}
curr_offset = 0
for line in f_bases_sizes:
    line = line.rstrip().split("\t")
    sizes[line[0]] = int(line[1])
    offsets[line[0]] = curr_offset
    curr_offset += int(line[1])
ipd_table = pd.read_csv(args.ipd, sep=",")
# fold_change_table = pd.read_csv("../data/original/JM083.fold-change.txt", sep=",")
f_fold_change = open(args.fold, "r")
fold_change_values = []
for line in f_fold_change:
    fold_change_values.append(float(line.rstrip()))
# Position,Strand,Base,Score,Trimmed Mean,Trimmed Error,Null Prediction,IPD Ratio,Coverage
# result = pd.DataFrame() # create empty dataframe
# for chromosome, size in sizes.items():
columns = ["Chromosome",
           "Position",
           "Base", 
           "Fold Change",
           "Top Score",
           "Top Trimmed Mean",
           "Top Trimmed Error",
           "Top Null Prediction",
           "Top IPD Ratio",
           "Top Coverage",
           "Bottom Score",
           "Bottom Trimmed Mean",
           "Bottom Trimmed Error",
           "Bottom Null Prediction",
           "Bottom IPD Ratio",
           "Bottom Coverage"]

def process(chromosome):
    print(chromosome)
    df = ipd_table[ipd_table["Chromosome"] == chromosome]
    result = {c: [] for c in columns}
    size = sizes[chromosome]
    for position in range(0, size):
        if position % (size // 20) == 0:
            print(f"{(position // (size // 20))}0% Done")
        # print(position / size)
        ipd_top = df[(df["Position"] == position) & (df["Strand"] == 0)]

        if ipd_top.empty or ipd_top.shape[0] != 1:
            ipd_top = None
        ipd_bottom = df[(df["Position"] == position) & (df["Strand"] == 1)]

        if ipd_bottom.empty or ipd_bottom.shape[0] != 1:
            ipd_bottom = None
        # result.iloc
        result["Position"].append(position)
        result["Base"].append(bases[chromosome][position])
        result["Chromosome"].append(chromosome)
        result["Fold Change"].append(fold_change_values[offsets[chromosome] + position])
        for column in df.columns:
            if column in ["Position", "Strand", "Base", "Chromosome"]:
                continue
            # if column == "Top Null Prediction" and ipd_top is not None:
            #     print("HERE BE NULL")
            #     print(type(ipd_top[column]))
            result["Top " + column].append(ipd_top[column].iloc[0] if ipd_top is not None else None)
            result["Bottom " + column].append(ipd_bottom[column].iloc[0] if ipd_bottom is not None else None)
        # print(result)
        result["Position"] = result["Position"].apply(lambda x: x - 1)
    return result


# Parallel
if __name__ == "__main__":
    if os.path.isfile(args.outfile):
        result = pd.read_csv(args.outfile, sep="\t")
    else:
        result = pd.DataFrame(columns=columns)
    # result = {c: [] for c in columns}
    if args.parallel:
        pool = Pool(os.cpu_count())
        res = pool.map(process, list(sizes.keys()))
        for r in res:
            result = pd.concat([result, pd.DataFrame.from_dict(res)], ignore_index=True)
            # for col in r:
            #     result[col].extend(res[col])
        # df = pd.DataFrame.from_dict(result)
        result.to_csv(args.outfile, sep="\t", index=False)
        pool.close()
        pool.join()
    else:
        print(result["Chromosome"].unique())
        print("======")
        print(list(sizes.keys()))
        for chromosome in list(sizes.keys()):
            print(chromosome)
            if chromosome in result["Chromosome"].unique():
                continue
            res = process(chromosome)
            # for col in res:
            #     result[col].extend(res[col])
            result = pd.concat([result, pd.DataFrame.from_dict(res)], ignore_index=True)
            # df = pd.DataFrame.from_dict(result)
            result.to_csv(args.outfile, sep="\t", index=False)



# Sequential



# row["IPD Top Ratio"] = ipd_top["IPD Ratio"] if not ipd_top.empty() else None
# row["IPD Bottom Ratio"] = ipd_bottom["IPD Ratio"] if not ipd_bottom.empty() else None
# row["Top Score"] = ipd_top["Score"] if not ipd_top.empty() else None
# row["Top Trimmed Mean"] = ipd_top["Trimmed Mean"] if not ipd_top.empty() else None
# row["Top Trimmed Error"] = ipd_top["Trimmed Error"] if not ipd_top.empty() else None
# row["Top Null Prediction"] = ipd_top["Null Prediction"] if not ipd_top.empty() else None
# Position,Strand,Base,Score,Trimmed Mean,Trimmed Error,Null Prediction,IPD Ratio,Coverage


# fold_change = fold_change_table["Fold Change"].repeat(2)
# print(fold_change.describe())
# print(fold_change.size)
# ipd_table = pd.read_csv("../data/original/ipd-h5.txt", sep=",")
# print(ipd_table.describe())
# print(ipd_table.shape[0])
# print(ipd_table.head(500))
# cols = ipd_table.columns + ["Fold Change"]
# table = pd.concat([ipd_table.reset_index(drop=True), fold_change.to_frame("Fold Change").reset_index(drop=True)], axis=1,ignore_index=True)
# print(table.describe())
# print(table.head(500))
# table.columns = cols
# print(table.columns)
# # table = table.drop(["Base"])
# for row in table.iterrows():
#     print(row["Base"])
#     if row["Base"] not in ["T", "A", "C", "G"]:
#         row["Base"] = bases[row["Chromosome"]][row["Position"]]



# Previous version:






# =================== OLD ========================

# import pandas as pd
# from multiprocessing import Pool, Lock
# import numpy as np 
# import os

# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument("-bases", "--bases", required=True)
# parser.add_argument("-sizes", "--sizes", required=True)
# parser.add_argument("-ipd", "--ipd", required=True)
# parser.add_argument("-fold", "--fold", required=True)
# parser.add_argument("-o", "--outfile", required=True)
# parser.add_argument("-parallel", "--parallel", action="store_true")
# args = parser.parse_args()

# f_bases = open(args.bases, "r")
# # f_bases_sizes = open("../data/original/LtaP_PB.sizes.txt", "r")
# bases = {}
# c_chromosome = None
# for line in f_bases:
#     line = line.rstrip()
#     if line[0] == '>':
#         c_chromosome = line[1:]
#         bases[c_chromosome] = []
#     else:
#         bases[c_chromosome].extend(list(line))
# f_bases_sizes = open(args.sizes, "r")
# sizes = {}
# offsets = {}
# curr_offset = 0
# for line in f_bases_sizes:
#     line = line.rstrip().split("\t")
#     sizes[line[0]] = int(line[1])
#     offsets[line[0]] = curr_offset
#     curr_offset += int(line[1])
# ipd_table = pd.read_csv(args.ipd, sep=",")
# # fold_change_table = pd.read_csv("../data/original/JM083.fold-change.txt", sep=",")
# f_fold_change = open(args.fold, "r")
# fold_change_values = []
# for line in f_fold_change:
#     fold_change_values.append(float(line.rstrip()))
# # Position,Strand,Base,Score,Trimmed Mean,Trimmed Error,Null Prediction,IPD Ratio,Coverage
# # result = pd.DataFrame() # create empty dataframe
# # for chromosome, size in sizes.items():
# columns = ["Chromosome",
#            "Position",
#            "Base", 
#            "Fold Change",
#            "Top Score",
#            "Top Trimmed Mean",
#            "Top Trimmed Error",
#            "Top Null Prediction",
#            "Top IPD Ratio",
#            "Top Coverage",
#            "Bottom Score",
#            "Bottom Trimmed Mean",
#            "Bottom Trimmed Error",
#            "Bottom Null Prediction",
#            "Bottom IPD Ratio",
#            "Bottom Coverage"]

# def create_dataframe(chromosome):
#     print(chromosome)
#     size = sizes[chromosome]
#     df = ipd_table[ipd_table["Chromosome"] == chromosome]
#     result = pd.DataFrame(columns=columns) # create empty dataframe
#     for position in range(size):
#         print(position / size)
#         ipd_top = df[(df["Position"] == position) & (df["Strand"] == 0)]
#         if ipd_top.shape[0] != 1:
#             ipd_top = None
#         ipd_bottom = df[(df["Position"] == position) & (df["Strand"] == 1)]
#         if ipd_bottom.shape[0] != 1:
#             ipd_bottom = None
#         # result.iloc
#         row = {}
#         row["Position"] = position
#         row["Base"] = bases[chromosome][position]
#         row["Chromosome"] = chromosome
#         row["Fold Change"] = fold_change_values[offsets[chromosome] + position]
#         if ipd_top is not None:
#             for col in ipd_top.columns:
#                 print(type(ipd_top[col].iloc[0]))
#                 # print(len(ipd_top[col].iloc[]))
#         for column in df.columns:
#             if column in ["Position", "Strand", "Base", "Chromosome"]:
#                 continue
#             row["Top " + column] = ipd_top[column].iloc[0] if ipd_top is not None else None
#             row["Bottom " + column] = ipd_bottom[column].iloc[0] if ipd_bottom is not None else None
#         result = result.append(pd.Series(row), ignore_index=True)
#         print(result.iloc[-1].tolist())
#     return result


# # Parallel
# if __name__ == "__main__":
#     if args.parallel:
#         pool = Pool(os.cpu_count())
#         result = pd.concat(pool.map(create_dataframe, list(sizes.keys())), ignore_index=True)
#         result.to_csv(args.outfile, sep="\t", index=False)
#         pool.close()
#         pool.join()
#     else:
#         result = []
#         for chromosome in list(sizes.keys()):
#             print(chromosome)
#             result.append(create_dataframe(chromosome))
#             df = pd.concat(result, ignore_index=True)
#             result.to_csv(args.outfile, sep="\t", index=False)
#         # result = pd.concat([create_dataframe(chromosome) for chromosome in list(sizes.keys())],ignore_index=True)
#         # result.to_csv(args.outfile, sep="\t", index=False)
