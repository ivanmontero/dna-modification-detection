import pandas as pd
import numpy as np 
import os
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-c", "--centers", help="The file containing center files.")
parser.add_argument("-s", "--sequences", help="The file containing sequences")
parser.add_argument("-o", "--outdir", help="The directory to hold output.")
args = parser.parse_args()

sequences = pd.read_csv(args.sequences)
centers = pd.read_csv(args.centers)

print(sequences.describe())
print(centers.describe())