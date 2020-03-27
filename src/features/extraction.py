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

    parser.add_argument(
        '-od', 
        '--outdir', 
        default = None,
        help = 'Output directory.')
    
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
    interval = 500
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

        if i % interval == 0:
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

    with counter.get_lock():
        counter.value += (i % interval)
        progress.n = counter.value
        progress.refresh()

    vectors = np.array(vectors)
    nbytes = vectors.nbytes
    splits = np.ceil(nbytes/1e9)
    chunks = np.array_split(vectors, splits)

    for i in range(len(chunks)):
        queue.put((position, i, chunks[i]))

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

def combine(results, window):
    combined_result = {}
    for i in range(multiprocessing.cpu_count()):
        combined_result[i] = []

    for item in results:
        process = item[0]
        combined_result[process].append(item[1:])

    results = []
    for key in combined_result:
        process = combined_result[key]
        ordered_list = [0] * len(process)

        for item in process:
            order = item[0]
            ordered_list[order] = item[1]

        results.append(np.vstack(ordered_list))

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

    # Get the folder path for the data folders.
    project_folder = utils.project_path(arguments.outdir)
    data_folder = os.path.join(project_folder, 'data')
    interm_folder = os.path.join(data_folder, 'interm')
    os.makedirs(interm_folder, exist_ok=True)

    # Chunk the dataset into the number of processors.
    data, total = chunking(
        data[arguments.columns],
        interm_folder,
        arguments.window)
    utils.end_time(start)

    # Setup the progress bar.
    start = utils.start_time(f'Using {len(data)} Cores')
    counter = multiprocessing.Value('i', 0)
    queue = multiprocessing.SimpleQueue()
    progress = tqdm.tqdm(total = total, unit = ' rows', leave = False)

    # Send off one job for each chunk.
    processes = []
    for chunk in data:
        process = multiprocessing.Process(target = windows, args = (chunk, queue, counter, progress))
        process.start()
        processes.append(process)

    results = []
    while multiprocessing.active_children():
        results.append(queue.get())

    utils.end_time(start)

    start = utils.start_time('Combining Data')
    results = combine(results, arguments.window)
    utils.end_time(start)

    # Save the feature vectors as numpy arrays.
    start = utils.start_time('Saving Data')
    processed_folder = os.path.join(data_folder, 'processed')
    os.makedirs(processed_folder, exist_ok=True)
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

