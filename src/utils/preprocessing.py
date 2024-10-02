import pandas as pd
import numpy as np
from Bio import SeqIO
import multiprocessing
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--processes', type=int, default=24, required=False)
args = parser.parse_args()

PROCESSES = args.processes
CHRS = [f"chr{x}" for x in list(range(1,23))]

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def process_data(chr, seq):
    
    path = f'{DIR_PATH}/../data/raw/FractionalMethylation/{chr}.fm'
    df = pd.read_csv(path, delimiter="\t", header=None)
    df.replace(-1, np.nan, inplace=True)


    df = df[df.isna().sum(axis=1) < 37]
    
    loc = df[0]
    percents = df.loc[:, df.columns != 0]
    
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

    percents = percents.apply(remove_outliers, axis=1)

    df = pd.DataFrame(np.array([loc, np.nanstd(percents, axis=1), np.nanmean(percents, axis=1)]).T)


    
    df = df.rename({0: "location", 1: "deviation", 2: "mean"}, axis=1).astype({"location":int})

    df = df[df["deviation"] < 0.1]
    with open(f"{DIR_PATH}/../data/{chr}.fasta", "w") as out: 
        for row in df.itertuples():
            sequence = str(seq[row.location-1280:row.location+1280+1])
            if "N" not in sequence:
                out.write(f"> {row.mean}\n")
                out.write(sequence + "\n")

    print("Finished", chr)


with open("/mnt/labshare/share/reference_genome/human/hg19/hg19.fa", "r") as fasta:
    pool = multiprocessing.Pool(processes=PROCESSES)

    for chr in SeqIO.parse(fasta, "fasta"):
        if chr.id not in CHRS:
            continue
            
        pool.apply_async(process_data, args=(chr.id, chr.seq))

    pool.close()
    pool.join()


