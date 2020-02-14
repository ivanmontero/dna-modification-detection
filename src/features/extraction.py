import multiprocessing
from tqdm import tqdm
import pandas as pd
import argparse
import time
import json

# Add Helper Functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import utils

# Return argparse arguments for extracting features from the data.
def setup():
    parser = argparse.ArgumentParser(
        description = 'Create a numpy objects with a set of feature vectors.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.version = 0.3

    parser.add_argument(
        '-i', 
        '--infile', 
        required = True,
        help = 'Input file.')

    parser.add_argument(
        '-w', 
        '--window', 
        default = 51,
        type = int, 
        help = 'The size of the window used for predictions.')

    parser.add_argument(
        '--columns',
        default = ['top_A',
                   'top_T',
                   'top_C',
                   'top_G', 
                   'top_ipd', 
                   'bottom_ipd'],
        help = 'List of columns to include as features.')

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

# Get path to project directory.
def project_path():
    script_path = os.path.abspath(__file__)
    script_folder = os.path.dirname(script_path)
    src_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(src_folder)
    
    return project_folder

def windows(data, window, progress):
    radius = int((window - 1)/2)

    if progress:
        bar = tqdm(total = len(data))

    vectors = [None]*len(data.index)
    for j in range(radius, len(data.index) - radius - 1):
        start = j - radius
        end = j + radius + 1
        section = data.iloc[start:end]

        start_position = data.index[start][1]
        end_position = data.index[end][1]

        # Checks if rows are contiguous.
        if (abs(end_position - start_position) > window):
        	continue
   
        vector = section.to_numpy().flatten(order = 'F')
        vectors[j] = vector

        if progress:
            bar.update()

    vectors = pd.DataFrame({'vectors': vectors}, index = data.index)

    if progress:
        bar.close()

    return vectors

def chunking(data, window):
    radius = int((window - 1)/2)
    num_processes = multiprocessing.cpu_count()
    chunk_size = int(len(data)/num_processes) + 1

    chunks = []
    for i in range(0, len(data), chunk_size):
        start = max(i - radius, 0)
        end = i + chunk_size + radius + 1

        chunks.append([data.iloc[start:end], window, False])

    chunks[0][2] = True
    return chunks

def combine(data, results, window):
    radius = int((window - 1)/2)
    combined_results = results[0].iloc[:-(radius + 1)]

    for i in range(1, len(results) - 1):
        trimmed = results[i].iloc[radius:-(radius + 1)]
        combined_results = pd.concat([combined_results, trimmed])

    trimmed = results[-1].iloc[radius:]
    combined_results = pd.concat([combined_results, trimmed])

    return pd.concat([data, combined_results], axis = 1)

def main():
    arguments = setup()
    data = pd.read_hdf(arguments.infile)
    chunks = chunking(data[arguments.columns], arguments.window)

    start = utils.start_time('Extracting Windows')
    with multiprocessing.Pool() as pool:
        results = pool.starmap(windows, chunks)
    utils.end_time(start)
    
    print ('Writing output.')
    data = combine(data, results, arguments.window)

    project_folder = utils.project_path()
    data_folder = os.path.join(project_folder, 'data')
    processed_folder = os.path.join(data_folder, 'processed')
    if arguments.prefix:
        filename = os.path.join(processed_folder, f'{arguments.prefix}_data.h5')
    else:
        filename = os.path.join(processed_folder, 'data.h5')

    data.to_hdf(filename, 'data', format = 'table')

    column_labels = []
    for column in arguments.columns:
        column_labels += [column] * arguments.window

    metadata = {'columns': column_labels,
            'arguments': vars(arguments)}

    if arguments.prefix:
        filename = os.path.join(processed_folder, f'{arguments.prefix}_metadata.json')
    else:
        filename = os.path.join(processed_folder, 'metadata.json')

    with open(filename, 'w') as outfile:
        json.dump(metadata, outfile, indent = 4)

if __name__ == '__main__':
    main()

