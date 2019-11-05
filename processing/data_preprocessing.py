from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import argparse
import time

# Return argparse arguments. 
def setup():
    parser = argparse.ArgumentParser(
        description = 'Create a HDF file with a set of peaks.')

    parser.version = 0.2

    parser.add_argument(
        '-i', 
        '--ipd', 
        required = True,
        help = 'IPD ratios table.')

    parser.add_argument(
        '-c', 
        '--fold-change',
        default = None, 
        help = 'Fold change file.')

    parser.add_argument(
        '-o', 
        '--output', 
        default = 'output.tsv',
        help = 'Output file.')
    
    return parser.parse_args()

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
    total_start = time.time()
    arguments = setup()

    if arguments.fold_change:
        print ('Reading fold change file.')
        start = time.time()
        fold_change = pd.read_csv(arguments.fold_change)

        # Set Multindex 
        fold_change = fold_change.set_index(['chromosome', 'position'])
        elapsed = time.time() - start
        print (f'{elapsed:.0f} seconds elapsed.')
    
    print ('Reading IPD file.')
    start = time.time()
    ipd = pd.read_csv(arguments.ipd)
    top_strand = ipd[ipd['strand'] == 0].drop(columns = ['strand'])
    bottom_strand = ipd[ipd['strand'] == 1].drop(columns = ['strand'])

    # Rename columns so they are unique for top and bottom strand. 
    top_strand = top_strand.rename(columns = {
        'refName': 'chromosome',
        'tpl': 'position',
        'base': 'top_base',
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
        'base': 'bottom_base',
        'score': 'bottom_score',
        'tMean': 'bottom_mean',
        'tErr': 'bottom_error',
        'modelPrediction': 'bottom_prediction',
        'ipdRatio': 'bottom_ipd',
        'coverage': 'bottom_coverage'
    })

    # Set multindex.
    top_strand = top_strand.set_index(['chromosome', 'position'])
    bottom_strand = bottom_strand.set_index(['chromosome', 'position'])
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

    print ('Encoding bases.')
    start = time.time()
    top_encoding = pd.get_dummies(top_strand['top_base'], prefix = 'top')
    bottom_encoding = pd.get_dummies(bottom_strand['bottom_base'], prefix = 'bottom')
   
    # Merge encodings.
    top_strand = pd.merge(top_strand, top_encoding, on = ['chromosome', 'position'])
    bottom_strand = pd.merge(bottom_strand, bottom_encoding, on = ['chromosome', 'position'])

    # Drop base column.
    top_strand = top_strand.drop(columns = 'top_base')
    bottom_strand = bottom_strand.drop(columns = 'bottom_base')
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

    print ('Merging files.')
    start = time.time()
    data = pd.merge(top_strand, bottom_strand, on = ['chromosome', 'position'])

    if arguments.fold_change:
        data = pd.merge(data, fold_change, on = ['chromosome', 'position'])
        
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.') 

    print ('Plotting histograms.')
    start = time.time()
    filename = '.'.join(arguments.output.split('.')[:-1]) + str('.pdf')
    plot(data, filename)
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

    print ('Normalizing data.')
    start = time.time()
    data = normalize(data)
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

    print ('Writing output.')
    start = time.time()
    data.round(4).to_hdf(arguments.output, 'data')
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')
    
    elapsed = time.time() - total_start
    print (f'{elapsed:.0f} seconds elapsed in total.')

if __name__ == '__main__':
    main()

