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

    parser.version = 0.4

    parser.add_argument(
        '-i', 
        '--ipd', 
        required = True,
        help = 'IPD ratios table.')

    parser.add_argument(
        '-f', 
        '--fold-change',
        default = False, 
        help = 'Fold change file.')

    parser.add_argument(
        '-p', 
        '--prefix', 
        default = False,
        help = 'Output prefix.')

    return parser.parse_args()

# Start the timer. 
def start_time(string = None):
    if string:
        print (string)
    return time.time()

# End the timer. 
def end_time(start, stop = False):
    seconds = time.time() - start
    hours, seconds =  seconds // 3600, seconds % 3600
    minutes, seconds = seconds // 60, seconds % 60
    string = f'{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}'
    if stop:
        return string
    print (f'{string} elapsed.')

# Return path to project level. 
def project_path():
    script_path = os.path.abspath(__file__)
    script_folder = os.path.dirname(script_path)
    src_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(src_folder)
    
    return project_folder

# Plot histograms of the data. 
def plot(data, filename, fold_change):
   
    columns = [
        ('fold_change', 'Fold Change', np.linspace(0, 20, 101)),
        ('top_ipd', 'Top IPD', np.linspace(0, 5, 101)),
        ('bottom_ipd', 'Bottom IPD', np.linspace(0, 5, 101)),
        ('top_coverage', 'Top Coverage', np.linspace(0, 1000, 101)),
        ('bottom_coverage', 'Bottom Coverage', np.linspace(0, 1000, 101)),
        ('top_score', 'Top Score', np.linspace(0, 100, 101)),
        ('bottom_score', 'Bottom Score', np.linspace(0, 100, 101)),
        ('top_mean', 'Top Mean', np.linspace(0, 10, 101)),
        ('bottom_mean', 'Bottom Mean', np.linspace(0, 10, 101)),
        ('top_error', 'Top Error', np.linspace(0, 2, 101)),
        ('bottom_error', 'Bottom Error', np.linspace(0, 2, 101))
    ]

    with PdfPages(filename) as pdf: 
        for column in columns:
            if (not fold_change) and (column[0] == 'fold_change'):
                continue
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

    columns = [
        'top_ipd',
        'bottom_ipd',
        'top_coverage',
        'bottom_coverage',
        'top_score',
        'bottom_score',
        'top_mean',
        'bottom_mean',
        'top_error',
        'bottom_error'
    ]

    for column in columns:
        data[column] = (data[column] - data[column].mean())/data[column].std()

    return data

def main():
    total_start = start_time()
    # Get argparse arguments. 
    arguments = setup()
    
    # If there is ChIP data: 
    if arguments.fold_change:
        start = start_time('Reading fold change file.')
        fold_change = pd.read_csv(
            arguments.fold_change,
            dtype = {
                'chromosome': 'category'
                })

        # Set Multindex. 
        fold_change.set_index([
            'chromosome', 
            'position'
            ], 
            inplace = True)
        end_time(start)

    # Load the IPD data. 
    start = start_time('Reading IPD file.')
    ipd = pd.read_csv(
        arguments.ipd, 
        usecols = [
            'ref_name',
            'index',
            'base',
            'strand',
            'score',
            'trimmed_mean',
            'trimmed_error',
            'ipd_ratio',
            'case_coverage'
        ], 
        dtype = {
            'ref_name': 'category',
            'base': 'category'
        })
    ipd.rename(
        columns = {
            'ref_name': 'chromosome', 
            'index': 'position',
        },
        inplace = True)
    ipd.set_index([
            'chromosome', 
            'position'
        ],
        inplace = True)

    top_strand = ipd[ipd['strand'] == 0].drop(columns = ['strand'])
    top_strand = top_strand.loc[~top_strand.index.duplicated()]
    bottom_strand = ipd[ipd['strand'] == 1].drop(columns = ['strand'])
    bottom_strand = bottom_strand.loc[~bottom_strand.index.duplicated()]

    # Rename columns so they are unique for top and bottom strand. 
    top_strand.rename(
        columns = {
            'base': 'top_base',
            'score': 'top_score',
            'trimmed_mean': 'top_mean',
            'trimmed_error': 'top_error',
            'ipd_ratio': 'top_ipd',
            'case_coverage': 'top_coverage'
        },
        inplace = True)

    bottom_strand.rename(
        columns = {
            'base': 'bottom_base',
            'score': 'bottom_score',
            'trimmed_mean': 'bottom_mean',
            'trimmed_error': 'bottom_error',
            'ipd_ratio': 'bottom_ipd',
            'case_coverage': 'bottom_coverage'
        },
        inplace = True)
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
    ipd = pd.merge(top_strand, bottom_strand, on = ['chromosome', 'position'])
    if arguments.fold_change:
        ipd = pd.merge(ipd, fold_change, on = ['chromosome', 'position'])
    end_time(start)

    # Plot the histograms of each feature. 
    start = start_time('Plotting histograms.')
    project_folder = project_path()
    reports_folder = os.path.join(project_folder, 'reports')

    if arguments.prefix:
        filename = os.path.join(reports_folder, f'{arguments.prefix}_histograms.pdf')
    else:
        filename = os.path.join(reports_folder, 'histograms.pdf')

    plot(ipd, filename, arguments.fold_change)
    end_time(start)

    # Normalize ipd and output to HDF file. 
    start = start_time('Writing output.')
    ipd = normalize(ipd).round(4)

    data_folder = os.path.join(project_folder, 'data')
    interm_folder = os.path.join(data_folder, 'interm')
    if arguments.prefix:
        filename = os.path.join(interm_folder, f'{arguments.prefix}_data.h5')
    else:
        filename = os.path.join(interm_folder, 'data.h5')

    ipd.to_hdf(filename, 'data', format = 'table')
    end_time(start)
    total_time = end_time(total_start, True)
    print (f'{total_time} elapsed in total.')

if __name__ == '__main__':
    main()
