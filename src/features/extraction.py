import multiprocessing
import pandas as pd
import numpy as np
import argparse
import shutil
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

# Extract windows around each base and return a dataframe.
def windows(data, output, index, positions, window, counter, progress):
    # Update progress bar after this many rows. 
    interval = 500

    n = 0
    for start in index:
        n += 1

        # Find the window.
        center = start + window//2
        end = start + window
        section = data[start:end]

        start_position = positions[start]
        end_position = positions[end]

        # Checks if rows are contiguous.
        if (abs(end_position - start_position) > window):
            section = np.zeros(section.shape[1] * window)
        else:
            section = section.flatten(order = 'F')
        
        # Flattens the section of the table using column first Fortran method.
        output[center] = section

        if n % interval == 0:
            with counter.get_lock():
                counter.value += interval
                progress.n = counter.value
                progress.refresh()

    with counter.get_lock():
        counter.value += (n % interval)
        progress.n = counter.value
        progress.refresh()

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
    os.makedirs(interm_folder, exist_ok = True)

    processed_folder = os.path.join(data_folder, 'processed')
    os.makedirs(processed_folder, exist_ok = True)
    if arguments.prefix:
        filename = os.path.join(
            processed_folder,
            f'{arguments.prefix}_data.npy')
    else:
        filename = os.path.join(
            processed_folder, 
            'data.npy')
    
    # Get the columns needed for extraction.
    positions = data.index.get_level_values('position').to_numpy()
    data = data[arguments.columns].to_numpy()
    rows, cols = data.shape
    radius = arguments.window//2

    # Create a memmap for the data and results.
    temp_filename = os.path.join(interm_folder, 'data.npy')
    array = np.memmap(
        filename = temp_filename,
        dtype = 'float32',
        mode = 'w+',
        shape = data.shape)
    output = np.memmap(filename = filename,
        dtype = 'float32',
        mode = 'w+',
        shape = (rows, cols * arguments.window))
    array[:,:] = data[:,:]
    output[:radius,:] = 0
    output[-(radius+1):,:] = 0

    # Determine chunk indices.
    indices = np.array_split(
        np.arange(rows - arguments.window),
        os.cpu_count())

    # Setup the progress bar.
    start = utils.start_time(f'Using {os.cpu_count()} Cores')
    counter = multiprocessing.Value('i', 0)
    progress = tqdm.tqdm(total = rows - arguments.window, unit = ' rows', leave = False)

    # Send off one job for each chunk.
    processes = []
    for index in indices:
        process = multiprocessing.Process(target = windows, args = (array, output, index, positions, arguments.window, counter, progress))
        process.start()
        processes.append(process)

    # Wait till all the processes are finished. 
    for process in processes:
        process.join()
    utils.end_time(start)

    start = utils.start_time(f'Saving metadata')
    # Save the metadata for later. 
    column_labels = []
    for column in arguments.columns:
        column_labels += [column] * arguments.window

    metadata = {
        'columns': column_labels,
        'rows': output.shape[0],
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

    # Clean up temporary directory.
    del array
    del output
    os.remove(temp_filename)
    
if __name__ == '__main__':
    main()

