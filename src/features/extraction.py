import multiprocessing
import pandas as pd
import argparse
import time
import json
import tqdm
import shutil
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

# (data, result, window, i, positions[i], counter, progress) 
# Extract windows around each base and return a dataframe.
def windows(data, result, window, indices, positions, counter, progress, queue):
    # Update progress bar after this many rows. 
    interval = 500
    invalid_indices = []

    for i, start in enumerate(indices):
        # Find the window. 
        end = start + window
        section = data[start:end]

        start_position = positions[i]
        end_position = positions[i + window - 1]

        if i % interval == 0:
            with counter.get_lock():
                counter.value += interval
                progress.n = counter.value
                progress.refresh()

        # Checks if rows are contiguous.
        if (abs(end_position - start_position) > window):
            invalid_indices.append(start)
            continue
        
        # # Flattens the section of the table using column first Fortran method.
        result[start,:] = section.reshape(-1)

    with counter.get_lock():
        counter.value += (i % interval)
        progress.n = counter.value
        progress.refresh()
    
    queue.put(invalid_indices)
# (result, output, i, counter, progress, queue)
def write_to_disk(result, output, indices, invalid_indices, out_loc, counter, progress, queue):
    # Update progress bar after this many rows. 
    interval = 500

    for i in indices:
        if i not in invalid_indices:
            output[out_loc[i],:] = result[i,:]
        
        if i % interval == 0:
            with counter.get_lock():
                counter.value += interval
                progress.n = counter.value
                progress.refresh()


    with counter.get_lock():
        counter.value += (i % interval)
        progress.n = counter.value
        progress.refresh()
    
    queue.put("done!")

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

    # Create a memmap for the data and results
    np_array = data[arguments.columns].to_numpy()
    os.makedirs(interm_folder+"/tmp/", exist_ok=True)
    array = np.memmap(interm_folder+"/tmp/data.npy", dtype='float32', mode='w+', shape=np_array.shape)
    result = np.memmap(interm_folder+"/tmp/result.npy", dtype='float32', mode='w+', shape=(np_array.shape[0]-(arguments.window-1), np_array.shape[1]*arguments.window))
    array[:,:] = np_array[:,:]
    positions = data.index.get_level_values('position').to_numpy()


    # Determine chunks:
    indices = np.array_split(np.arange(result.shape[0]), os.cpu_count())

    # Setup the progress bar.
    start = utils.start_time(f'Using {os.cpu_count()} Cores')
    counter = multiprocessing.Value('i', 0)
    queue = multiprocessing.SimpleQueue()
    progress = tqdm.tqdm(total = result.shape[0], unit = ' rows', leave = False)

    # with multiprocessing.Pool(os.cpu_count()) as p:
    #     p.starmap(windows,[(data, result, arguments.window, i, positions[i], counter, progress) for i in indices])

    # Send off one job for each chunk.
    processes = []
    for i in indices:
        process = multiprocessing.Process(target = windows, args = (array, result, arguments.window, i, positions[i[0]:i[-1]+arguments.window], counter, progress, queue))
        process.start()
        processes.append(process)
    # for p in processes:
    #     p.join()

    # Get all the results back. 
    invalid_indices = set()
    while multiprocessing.active_children():
        while not queue.empty():
            invalid_indices.update(queue.get())
    utils.end_time(start)


    # Create the rsult file
    start = utils.start_time(f'Creating final output file')

    processed_folder = os.path.join(data_folder, 'processed')
    os.makedirs(processed_folder, exist_ok=True)
    if arguments.prefix:
        filename = os.path.join(processed_folder, f'{arguments.prefix}_data.npy')
    else:
        filename = os.path.join(processed_folder, 'data.npy')
    output = np.memmap(filename, dtype='float32', mode='w+', shape=(result.shape[0]-len(invalid_indices), result.shape[1]))

    valid_indices = np.ones(result.shape[0])
    valid_indices[list(invalid_indices)] = 0
    out_loc = (np.cumsum(valid_indices) - 1).astype(int).tolist()

    counter = multiprocessing.Value('i', 0)
    queue = multiprocessing.SimpleQueue()
    progress = tqdm.tqdm(total = result.shape[0], unit = ' rows', leave = False)
    processes = []
    for i in indices:
        process = multiprocessing.Process(target = write_to_disk, args = (result, output, i, invalid_indices, out_loc, counter, progress, queue))
        process.start()
        processes.append(process)

    # Get all the results back. 
    results = []
    while multiprocessing.active_children():
        while not queue.empty():
            results.append(queue.get())

    # The two alternatives that blow up memory
    # output[:,:] = result[np.delete(np.arange(result.shape[0]), invalid_indices),:]
    # output[:,:] = np.delete(result, invalid_indices, axis=0)
    utils.end_time(start)

    start = utils.start_time(f'Saving metadata')
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
    shutil.rmtree(f"{interm_folder}/tmp/")    
    
if __name__ == '__main__':
    main()

