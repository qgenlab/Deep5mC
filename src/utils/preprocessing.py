import pandas as pd
import numpy as np
from Bio import SeqIO
import multiprocessing
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("reference_genome", required=True, type=argparse.FileType('r', encoding='UTF-8'))
parser.add_argument('-p', '--processes', type=int, default=24, required=False)
args = parser.parse_args()

PROCESSES = args.processes
CHRS = [f"chr{x}" for x in list(range(1,23))]

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def process_data(chr, seq):
    # load file
    path = f'{DIR_PATH}/../data/raw/FractionalMethylation/{chr}.fm'
    df = pd.read_csv(path, delimiter="\t", header=None)

    # -1 is a missing value
    df.replace(-1, np.nan, inplace=True)

    # Remove rows that are only missing values
    df = df[df.isna().sum(axis=1) < 37]
    
    loc = df[0]
    percents = df.loc[:, df.columns != 0]

    # This function removes the smallest and largest two values from a row, as long as there are more than 4 values in that row
    def remove_outliers(row):
        row = row.to_numpy()
        
        if np.count_nonzero(~np.isnan(row)) <= 4:
            return pd.Series(row)
            
        smallest_i = -1
        smallest_i_2 = -1
        biggest_i = -1
        biggest_i_2 = -1
        for i, x in enumerate(row):
            if np.isnan(x):
                continue
            if x < row[smallest_i] or smallest_i == -1:
                smallest_i_2 = smallest_i
                smallest_i = i
            elif x < row[smallest_i_2] or smallest_i_2 == -1:
                smallest_i_2 = i
                
            if x > row[biggest_i] or biggest_i == -1:
                biggest_i_2 = biggest_i
                biggest_i = i
            elif x > row[biggest_i_2] or biggest_i_2 == -1:
                biggest_i_2 = i

        return pd.Series(np.delete(row, [smallest_i, smallest_i_2, biggest_i, biggest_i_2]))

    # Apply the above function to all rows
    percents = percents.apply(remove_outliers, axis=1)

    # Calculate the mean and standard deviation, ignoring NaN values
    df = pd.DataFrame(np.array([loc, np.nanstd(percents, axis=1), np.nanmean(percents, axis=1)]).T)

    # Reformat data
    df = df.rename({0: "location", 1: "deviation", 2: "mean"}, axis=1).astype({"location":int})

    # Filter by standard deviation
    df = df[df["deviation"] < 0.1]

    # Write percent and sequence to file
    with open(f"{DIR_PATH}/../data/{chr}.fasta", "w") as out: 
        for row in df.itertuples():
            sequence = str(seq[row.location-1280:row.location+1280+1])
            if "N" not in sequence:
                out.write(f"> {row.mean}\n")
                out.write(sequence + "\n")

    print("Finished", chr)


# Read reference genome and begin processing data
with open(args.reference_genome.name, "r") as fasta:
    pool = multiprocessing.Pool(processes=PROCESSES)

    for chr in SeqIO.parse(fasta, "fasta"):
        if chr.id not in CHRS:
            continue
            
        pool.apply_async(process_data, args=(chr.id, chr.seq))

    pool.close()
    pool.join()


