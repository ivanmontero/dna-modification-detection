from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time
import os

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
        default = False, 
        help = 'Fold change file.')

    parser.add_argument(
        '-o', 
        '--output', 
        default = False,
        help = 'Output prefix.')

    return parser.parse_args()

# Start the timer. 
def start_time(string):
    print (string)
    return time.time()

# End the timer. 
def end_time(start):
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

def project_path():
    script_path = os.path.abspath(__file__)
    script_folder = os.path.dirname(script_path)
    src_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(src_folder)
    
    return project_folder

# Plot histograms of the data. 
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
               ('bottom_error', 'Bottom Error', np.linspace(0, 2, 101))]

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

# Mean and standard deviation normalization of the data. 
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
               'bottom_error']

    for column in columns:
        data[column] = (data[column] - data[column].mean())/data[column].std()

    return data

def main(): 
    # Get argparse arguments. 
    arguments = setup()
    
    # If there is ChIP data: 
    if arguments.fold_change:
        start = start_time('Reading fold change file.')
        fold_change = pd.read_csv(arguments.fold_change)

        # Set Multindex. 
        fold_change = fold_change.set_index(['chromosome', 'position'])
        end_time(start)

    # Load the IPD data. 
    start = start_time('Reading IPD file.')
    ipd = pd.read_csv(arguments.ipd)
    top_strand = ipd[ipd['strand'] == 0].drop(columns = ['strand'])
    bottom_strand = ipd[ipd['strand'] == 1].drop(columns = ['strand'])

    # Rename columns so they are unique for top and bottom strand. 
    top_strand = top_strand.rename(columns = {
        'ref_name': 'chromosome',
        'index': 'position',
        'base': 'top_base',
        'score': 'top_score',
        'trimmed_mean': 'top_mean',
        'trimmed_error': 'top_error',
        'ipd_ratio': 'top_ipd',
        'case_coverage': 'top_coverage'
    })

    bottom_strand = bottom_strand.rename(columns = {
        'ref_name': 'chromosome',
        'index': 'position',
        'base': 'bottom_base',
        'score': 'bottom_score',
        'trimmed_mean': 'bottom_mean',
        'trimmed_error': 'bottom_error',
        'ipd_ratio': 'bottom_ipd',
        'case_coverage': 'bottom_coverage'
    })

    # Set multindex.
    top_strand = top_strand.set_index(['chromosome', 'position'])
    bottom_strand = bottom_strand.set_index(['chromosome', 'position'])
    end_time(start)

    # Encode bases. 
    start = start_time('Encoding bases.')
    top_encoding = pd.get_dummies(top_strand['top_base'], prefix = 'top')
    bottom_encoding = pd.get_dummies(bottom_strand['bottom_base'], prefix = 'bottom')
   
    # Merge encodings.
    top_strand = pd.merge(top_strand, top_encoding, on = ['chromosome', 'position'])
    bottom_strand = pd.merge(bottom_strand, bottom_encoding, on = ['chromosome', 'position'])

    # Drop base column.
    top_strand = top_strand.drop(columns = 'top_base')
    bottom_strand = bottom_strand.drop(columns = 'bottom_base')
    end_time(start)

    # Merge the top strand and bottom strand and ChIP if present. 
    start = start_time('Merging files.')
    data = pd.merge(top_strand, bottom_strand, on = ['chromosome', 'position'])
    if arguments.fold_change:
        data = pd.merge(data, fold_change, on = ['chromosome', 'position'])
    end_time(start)

    # Plot the histograms of each feature. 
    start = start_time('Plotting histograms.')
    top_level = project_path()
    report_folder = os.path.join(top_level, 'reports')

    if arguments.output:
        filename = os.path.join(report_folder, f'{arguments.output}_histograms.pdf')
    else:
        filename = os.path.join(report_folder, 'histograms.pdf')

    plot(data, filename)
    end_time(start)

    # Normalize data and output to HDF file. 
    start = start_time('Writing output.')
    data = normalize(data).round(4)

    data_folder = os.path.join(top_level, 'data')
    interm_folder = os.path.join(data_folder, 'interm')
    if arguments.output:
        filename = os.path.join(interm_folder, f'{arguments.output}_merged_data.h5')
    else:
        filename = os.path.join(interm_folder, 'merged_data.h5')

    data.to_hdf(filename, 'data')
    end_time(start)

if __name__ == '__main__':
    main()
