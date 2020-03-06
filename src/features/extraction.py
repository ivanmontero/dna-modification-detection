import multiprocessing
import pandas as pd
import argparse
import time
import json
import tqdm

# Add Helper Functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import utils

# Return argparse arguments for extracting features from the data.
def setup():
    parser = argparse.ArgumentParser(
        description = 'Create a numpy objects with a set of feature vectors.', 
        formatter_class = argparse.RawTextHelpFormatter)

    parser.version = 0.3

    parser.add_argument(
        '-i', 
        '--infile', 
        required = True,
        help = 'Input file HDF file.')

    parser.add_argument(
        '-w', 
        '--window', 
        default = 51,
        type = int, 
        help = 'The size of the window used for predictions.\n(Default: 51)')

    parser.add_argument(
        '--columns',
        default = ['top_A',
                   'top_T',
                   'top_C',
                   'top_G', 
                   'top_ipd', 
                   'bottom_ipd'],
        help = 'List of columns to include as features.\n' + 
               '(Default: [top_A, top_T, top_C, top_G, top_ipd, bottom_ipd])')

    parser.add_argument(
        '-p', 
        '--prefix', 
        default = False,
        help = 'Output prefix.')

    # Turn off the progress bars when running on a cluster. 
    parser.add_argument(
        '--progress-off',
        default = False,
        action = 'store_true', 
        help = argparse.SUPPRESS)
    
    return parser.parse_args()

import numpy as np

# Extract windows around each base and return a dataframe.
def windows(chunk, queue, counter, progress):
    data = chunk['data']
    index = chunk['index']
    window = chunk['window']
    position = chunk['position']
    folder = chunk['folder']

    # Radius is half the window size rounded down. So 51 is 25. 
    radius = int((window - 1)/2)

    # Update progress bar after this many rows. 
    interval = 100
    columns = data.shape[1]
    vectors = [[0] * columns * window] * len(data)

    # Skip the first radius rows, and the last radius rows since we can't get
    # full windows there anyways.
    for i in range(radius, len(data) - radius - 1):
        # Find the window. 
        start = i - radius
        end = i + radius + 1
        section = data[start:end]

        start_position = index[start]
        end_position = index[end]

        if (i % interval == 0) & (progress):
            with counter.get_lock():
                counter.value += interval
                progress.n = counter.value
                progress.refresh()

        # Checks if rows are contiguous.
        if (abs(end_position - start_position) > window):
            continue
        
        # # Flattens the section of the table using column first Fortran method.
        vector = section.flatten(order = 'F')
        vectors[i] = vector

    if progress:
        with counter.get_lock():
            counter.value += (i % interval)
            progress.n = counter.value
            progress.refresh()

    filename = os.path.join(folder, f'{position}.npy')
    np.save(filename, np.array(vectors))

# Divide the table into the number of cores available so we can use the 
# multiprocessing package. 
def chunking(data, interm_folder, window):
    index = data.index.get_level_values('position').to_numpy()
    array = data.to_numpy()

    radius = int((window - 1)/2)
    # The chunk size is the number of rows assigned to each core. We never want 
    # to have left over rows so we add one. 
    num_processes = multiprocessing.cpu_count()
    chunk_size = int(len(data)/num_processes) + 1

    chunks = []
    position = 0
    total = 0
    for i in range(0, len(data), chunk_size):
        start = max(i - radius, 0)
        end = i + chunk_size + radius + 1

        section = array[start:end].copy()
        index_section = index[start:end].copy()

        total  += len(section) - radius - 2

        chunks.append({
            'data': section,
            'index': index_section,
            'window': window,
            'position': position,
            'folder': interm_folder,
        })
        position += 1

    return chunks, total

def combine(folder, window):
    # Read from disk.
    results = []
    for i in range(multiprocessing.cpu_count()):
        filename = os.path.join(folder, f'{i}.npy')
        results.append(np.load(filename))
        os.remove(filename)

    radius = int((window - 1)/2)
    first_result = results[0][:-(radius + 1)]
    combined_result = [first_result]

    for i in range(1, len(results) - 1):
        trimmed_result = results[i][radius:-(radius + 1)]
        combined_result.append(trimmed_result)
        
    final_result = results[-1][radius:]
    combined_result.append(final_result)
    combined_result = np.vstack(combined_result)
    valid = combined_result.any(axis = 1)
    
    return combined_result

def main():
    total_start = utils.start_time()
    arguments = setup()

    # Read the data and use only the necessary columns for creating feature
    # vectors.
    start = utils.start_time('Reading Data')
    data = pd.read_hdf(arguments.infile)

    project_folder = utils.project_path()
    data_folder = os.path.join(project_folder, 'data')
    interm_folder = os.path.join(data_folder, 'interm')

    chunks, total = chunking(data[arguments.columns], interm_folder, arguments.window)
    utils.end_time(start)

    # We then send a set number of jobs equal to the number of cores. Each
    # process loads its data into a queue and we dequeue as they come into it. 
    start = utils.start_time(f'Using {len(chunks)} Cores')
    counter = multiprocessing.Value('i', 0)
    queue = multiprocessing.Queue(maxsize = 0)
    progress = tqdm.tqdm(total = total, unit = ' rows', leave = False)
    
    processes = []
    for chunk in chunks:
        process = multiprocessing.Process(target = windows, args = (chunk, queue, counter, progress))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    utils.end_time(start)

    
    # Combine the results and write to disk. 
    start = utils.start_time('Combining Data')
    results = combine(interm_folder, arguments.window)

    # Save the data folder again.
    processed_folder = os.path.join(data_folder, 'processed')
    if arguments.prefix:
        filename = os.path.join(processed_folder, f'{arguments.prefix}_data.npy')
    else:
        filename = os.path.join(processed_folder, 'data.npy')
    np.save(filename, results)

    # Save the metadata for later. 
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

    utils.end_time(start)
    total_time = utils.end_time(total_start, True)
    print (f'{total_time} elapsed in total.')

if __name__ == '__main__':
    main()

