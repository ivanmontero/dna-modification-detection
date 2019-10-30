from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import argparse
import time

# Return argparse arguments. 
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
        '--output', 
        default = 'output.tsv',
        help = 'Output file.')
    
    return parser.parse_args()

# Read fasta file and output dictionary with chromosomes, positions, and base
# information. 
def read_fasta(filename):
    sequence = ''
    chromosomes = []
    position = []

    with open(filename) as infile:
        current = ''
        for line in infile:

            # Headers begin with the '>' symbol. 
            if '>' in line:
                if current:
                    chromosomes += [name]*len(current)
                    position += range(1,len(current)+1)
                    sequence += current

                name = line.strip()[1:]
                current = ''
            else:
                current += line.strip()

    chromosomes += [name]*len(current)
    position += range(1,len(current)+1)
    sequence += current

    return {'chromosome': chromosomes, 
            'position': position, 
            'top': list(sequence.upper())}

# Get the complement DNA sequence for information on the other strand.
def get_complement(sequence):
    mapping = {'A': 'T',
               'T': 'A',
               'C': 'G',
               'G': 'C',
               'N': 'N'}

    complement = []
    for base in sequence:
        complement.append(mapping[base])

    return complement

# Read in the file with fold change data. 
def read_fold_change(filename):
    fold_change = []   
    with open(filename) as infile:
        for line in infile:
            fold_change.append(float(line))

    return {'fold_change': fold_change}

# Unfortunately it looks like the data columns don't match in size. In
# particular, the fasta file has a total length of 32,226,625 bases, and the fold
# change file has 32,226,627 bases. We trim off the difference in this function. 
def balance_sizes(sequence, fold_change):
    sequence_size = len(sequence['chromosome'])
    fold_change_size = len(fold_change['fold_change'])

    if sequence_size > fold_change_size:
        sequence['chromosome'] = sequence['chromosome'][:fold_change_size]
        sequence['top'] = sequence['top'][:fold_change_size]
    elif fold_change_size > sequence_size:
        fold_change['fold_change'] = fold_change['fold_change'][:sequence_size]

    sequence['bottom'] = get_complement(sequence['top'])
    return pd.DataFrame(sequence), pd.DataFrame(fold_change)

def plot(data, filename):

    columns = [('fold_change', 'Fold Change', np.linspace(0, 20, 101)),
               ('top_ipd', 'Top IPD', np.linspace(0, 5, 101)),
               ('bottom_ipd', 'Bottom IPD', np.linspace(0, 5, 101)),
               ('top_coverage', 'Top Coverage', np.linspace(0, 1000, 101)),
               ('bottom_coverage', 'Bottom Coverage', np.linspace(0, 1000, 101)),
               ('top_score', 'Top Score', np.linspace(0, 100, 101)),
               ('bottom_score', 'Bottom Score', np.linspace(0, 100, 101)),
               ('top_mean', 'Top Mean', np.linspace(0, 10, 101)),
               ('bottom_mean', 'Bottom Mean', np.linspace(0, 10, 101)),
               ('top_error', 'Top Error', np.linspace(0, 2, 101)),
               ('bottom_error', 'Bottom Error', np.linspace(0, 2, 101)),
               ('top_prediction', 'Top Prediction', np.linspace(0, 5, 101)),
               ('bottom_prediction', 'Bottom Prediction', np.linspace(0, 5, 101))]

    with PdfPages(filename) as pdf: 
        for column in columns:
            index = column[0]
            name = column[1]
            bins = column[2]
            current = data[index][~pd.isnull(data[index])]

            plt.figure(figsize = (6,6), dpi = 100)
            plt.hist(current, bins = bins)
            plt.title(name)
            pdf.savefig()
            plt.close()

def normalize(data):

    columns = ['top_ipd',
               'bottom_ipd',
               'top_coverage',
               'bottom_coverage',
               'top_score',
               'bottom_score',
               'top_mean',
               'bottom_mean',
               'top_error',
               'bottom_error',
               'top_prediction',
               'bottom_prediction']

    for column in columns:
        data[column] = (data[column] - data[column].mean())/data[column].std()

    return data

def main():
    start = time.time()
    arguments = setup()

    print ('Reading FASTA file.')
    sequence = read_fasta(arguments.fasta)
    print ('Reading fold change file.')
    fold_change = read_fold_change(arguments.fold_change)
    sequence, fold_change = balance_sizes(sequence, fold_change)
    print ('Merging files.')
    data = sequence.join(fold_change)
    
    print ('Reading IPD file.')
    ipd = pd.read_csv(arguments.ipd)
    top_strand = ipd[ipd['strand'] == 0].drop(columns = ['strand', 'base'])
    bottom_strand = ipd[ipd['strand'] == 1].drop(columns = ['strand', 'base'])

    top_strand = top_strand.rename(columns = {
        'refName': 'chromosome',
        'tpl': 'position',
        'score': 'top_score',
        'tMean': 'top_mean',
        'tErr': 'top_error',
        'modelPrediction': 'top_prediction',
        'ipdRatio': 'top_ipd',
        'coverage': 'top_coverage'
    })

    bottom_strand = bottom_strand.rename(columns = {
        'refName': 'chromosome',
        'tpl': 'position',
        'score': 'bottom_score',
        'tMean': 'bottom_mean',
        'tErr': 'bottom_error',
        'modelPrediction': 'bottom_prediction',
        'ipdRatio': 'bottom_ipd',
        'coverage': 'bottom_coverage'
    })

    print ('Merging files.')
    data = pd.merge(data, top_strand, on = ['chromosome', 'position'], how = 'outer')
    data = pd.merge(data, bottom_strand, on = ['chromosome', 'position'], how = 'outer')

    print ('Plotting histograms.')
    filename = '.'.join(arguments.output.split('.')[:-1]) + str('.pdf')
    plot(data, filename)

    print ('Normalizing data.')
    data = normalize(data)

    print ('Encoding bases.')
    top_encoding = pd.get_dummies(data['top'])
    bottom_encoding = pd.get_dummies(data['bottom'])
    data = data.drop(columns = ['top', 'bottom'])
    data = data.join(top_encoding)
    data = data.join(bottom_encoding)

    print (data)

    print ('Writing output.')
    data.round(4).to_csv(arguments.output,
        index = False, 
        columns = ['chromosome',
                   'position',
                   'fold_change',
                   'top_A',
                   'top_T',
                   'top_C',
                   'top_G',
                   'bottom_A',
                   'bottom_T',
                   'bottom_C',
                   'bottom_G',
                   'top_ipd',
                   'bottom_ipd',
                   'top_coverage',
                   'bottom_coverage',
                   'top_score',
                   'bottom_score',
                   'top_mean',
                   'bottom_mean',
                   'top_error',
                   'bottom_error',
                   'top_prediction',
                   'bottom_prediction'])
    
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

if __name__ == '__main__':
    main()

