import data_extraction
import pandas as pd
import numpy as np
import json
import time
import os

def main():
    total_start = data_extraction.start_time()
    arguments = data_extraction.setup()

    start = data_extraction.start_time('Reading data.')
    data = pd.read_hdf(arguments.infile, columns =  arguments.columns)
    data_extraction.end_time(start)

    start = data_extraction.start_time('Extracting windows.')
    features, positions = data_extraction.windows(data.index.values, data, arguments.window, arguments.columns)
    data_extraction.end_time(start)

    column_labels = []
    for column in arguments.columns:
        column_labels += [column] * arguments.window

    print ('Writing output.')
    data = {
        'columns': column_labels,
        'vectors': features, 
        'positions': positions
    }

    project_folder = data_extraction.project_path()
    data_folder = os.path.join(project_folder, 'data')
    processed_folder = os.path.join(data_folder, 'processed')
    if arguments.prefix:
        filename = os.path.join(processed_folder, f'{arguments.prefix}_data.json')
    else:
        filename = os.path.join(processed_folder, 'data.json')

    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent = 4)

    total_time = data_extraction.end_time(total_start, True)
    print (f'{total_time} elapsed in total.')

if __name__ == '__main__':
    main()






