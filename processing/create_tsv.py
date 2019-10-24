import pandas as pd
import numpy as np 
import argparse
import os

def setup():
    parser = argparse.ArgumentParser(
        description = 'Create a TSV file with a set of peaks.')

    parser.version = 0.1

    parser.add_argument(
        '-f', 
        '--fasta', 
        required = True,
        help = 'Fasta file for genome.')

    parser.add_argument(
        '-i', 
        '--ipd', 
        required = True,
        help = 'IPD ratios table.')

    parser.add_argument(
        '-c', 
        '--fold-change', 
        required = True,
        help = 'Fold change file.')

    parser.add_argument(
        '-o', 
        '--outfile', 
        required = True,
        help = 'Outfile name.')
    
    return parser.parse_args()

# Read fasta file and output dictionary of {name: sequence}
def read_fasta(filename):
    sequence = ''
    chromosomes = []

    with open(filename) as infile:
        current = ''
        for line in infile:
            if '>' in line:
                if current:
                    chromosomes += [name]*len(current)
                    sequence += current

                name = line.strip()[1:]
                current = ''
            else:
                current += line.strip()

    chromosomes += [name]*len(current)
    sequence += current

    return {'chromosome': chromosomes, 'base': list(sequence.upper())}

def read_fold_change(filename):
    fold_change = []   
    with open(filename) as infile:
        for line in infile:
            fold_change.append(float(line))

    return {'fold_change': fold_change}

# Unfortunately it looks like the data columns don't match in size. In
# particular, the fasta file has a total length of XXXX bases, and the fold
# change file has XXXX bases. We trim off the difference in this function. 
def balance_sizes(sequence, fold_change):
    sequence_size = len(sequence['chromosome'])
    fold_change_size = len(fold_change['fold_change'])

    print (sequence_size, fold_change_size)
    if sequence_size > fold_change_size:
        sequence['chromosome'] = sequence['chromosome'][:fold_change_size]
        sequence['base'] = sequence['base'][:fold_change_size]
    elif fold_change_size > sequence_size:
        fold_change['fold_change'] = fold_change['fold_change'][:sequence_size]

    return pd.DataFrame(sequence), pd.DataFrame(fold_change)

def main():
    # arguments = setup()
    sequence = read_fasta('../Data/LtaP_PB.genome.fasta')
    fold_change = read_fold_change('../Data/JM083.fold-change.txt')
    sequence, fold_change = balance_sizes(sequence, fold_change)

    new_data = sequence.join(fold_change)
    print (new_data)


if __name__ == '__main__':
    main()

