import data_extraction
import multiprocessing
import pandas as pd
import time

def windows(data):
    chromosomes = {}
    for i in set(data.index.get_level_values('chromosome')):
        chromosomes[i] = data.xs(i, drop_level = False)

    dataframes = []
    for i in chromosomes:
        current = chromosomes[i]
        index = current.index
        vectors = []
        lengths = []
        for j in range(len(index)):
            start = j - 25
            end = j + 25
            section = data.iloc[start:end]
            vector = section.to_numpy().flatten(order = 'F')
            if len(vector) < 51:
                vector = None
            vectors.append(vector)

        vectors = pd.DataFrame({'vectors': vectors}, index = current.index)
        dataframes.append(vectors)

    return pd.concat(dataframes)

def main():
    arguments = data_extraction.setup()
    data = pd.read_hdf(arguments.infile, columns = arguments.columns)

    num_processes = multiprocessing.cpu_count()
    chunk_size = int(len(data)/num_processes) + 1
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data.iloc[i:i + chunk_size])

    start = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(windows, chunks)
        vectors = pd.concat(results)
    end = time.time()
    print(f'Multiprocessing: {end - start}')

    print (vectors)



if __name__ == '__main__':
    main()

