import multiprocessing
from tqdm import tqdm
import pandas as pd
import argparse
import time

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
        '--ipd',
        default = 2,
        type = float,
        help = 'IPD threshold value.')

    parser.add_argument(
        '--fold-change',
        default = 10,
        type = float, 
        help = 'Fold change threshold value.')

    parser.add_argument(
        '-e',
        '--examples',
        default = 1000,
        type = int,
        help = 'Max number of examples from each class.')

    parser.add_argument(
        '--save-classes',
        action='store_true',
        default = False,
        help = argparse.SUPPRESS)

    parser.add_argument(
        '-p', 
        '--prefix', 
        default = False,
        help = 'Output prefix.')

    parser.add_argument(
        '-excl', 
        '--exclude',
        nargs='+',
        default=[],
        help = 'List of chromosomes to exclude processing')
    
    parser.add_argument(
        '-incl', 
        '--include',
        nargs='+',
        default=[],
        help = 'List of chromosomes to only process')
    
    parser.add_argument(
        '-c',
        '--center',
        default=False,
        action='store_true',
        help = 'Whether to only center on As and Ts'
    )
    
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

    chromosomes = {}
    for name in set(data.index.get_level_values('chromosome')):
        chromosomes[name] = data.xs(name, drop_level = False)

    if progress:
        bar = tqdm(total = len(data))
    dataframes = []
    for name in chromosomes:
        dataset = chromosomes[name]

        vectors = []
        for j in range(len(dataset.index)):
            start = j - radius
            end = j + radius + 1
            section = dataset.iloc[start:end]

            if len(section) < window:
                vector = None
            else:
                vector = section.to_numpy().flatten(order = 'F')
            vectors.append(vector)

            if progress:
                bar.update()

        vectors = pd.DataFrame({'vectors': vectors}, index = dataset.index)
        dataframes.append(vectors)

    if progress:
        bar.close()

    return pd.concat(dataframes)

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

def combine(data, results):
    for chunk in results:
        data = data.combine_first(chunk)

    return data

def main():
    arguments = setup()
    data = pd.read_hdf(arguments.infile)
    chunks = chunking(data[arguments.columns], arguments.window)

    start = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.starmap(windows, chunks)
    end = time.time()
    print(f'Multiprocessing: {end - start}')
    
    start = time.time()
    data = combine(data, results)
    end = time.time()
    print(f'Join: {end - start}')


if __name__ == '__main__':
    main()

