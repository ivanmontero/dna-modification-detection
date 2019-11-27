import data_extraction
import pandas as pd
import numpy as np
import json
import time
import os

def main():
    total_start = time.time()
    arguments = data_extraction.setup()

    print ('Reading data.')
    start = time.time()
    data = pd.read_hdf(arguments.infile)
    data = data[arguments.columns]
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

    print ('Extracting windows.')
    start = time.time()
    features, positions = data_extraction.windows(data, data, arguments.window, arguments.columns)
    elapsed = time.time() - start
    print (f'{elapsed:.0f} seconds elapsed.')

    print ('Writing output.')
    data = {'vectors': features.tolist(), 
            'positions': positions.tolist()}

    if arguments.output:
        with open(arguments.output, 'w') as outfile:
            json.dump(data, outfile)
    else:
        directory = os.path.dirname(arguments.infile)
        filename = os.path.join(directory, 'data.json')
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)

    elapsed = time.time() - total_start
    print (f'{elapsed:.0f} seconds elapsed in total.')

if __name__ == '__main__':
    main()






