from collections import deque

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import numpy as np
from Bio import SeqIO

from . import GlobalParameters

from tqdm import tqdm

token_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, 'A': 3, 'C': 4, 'G': 5, 'T': 6, 'OTHER': 7}

# Dataset returns a percent (expected result), followed by the one-hot encoded sequence. 
# The file itself has a character string for the sequence, but MethylationDataset will convert it in the encode() function.
class MethylationDataset(Dataset):
    def __init__(self, data_file, training):

        self.percents = deque()
        self.sequences = deque()

        # get input data count
        with open(data_file) as file:
            length = "".join(file.readlines()).count(">")
        
        # turn .fasta format into two numpy arrays
        for i, record in tqdm(enumerate(SeqIO.parse(data_file, "fasta")), total=length, ncols=GlobalParameters.ncols):
            self.percents.extend([float(record.name)])
            self.sequences.extend([self.encode(record.seq)])
            
        self.percents = np.array(self.percents)
        self.sequences = np.array(self.sequences)

        # balance data
        if training and GlobalParameters.percent_unmethylated != 0:
            # Returns indicies where data is positive
            positive = np.where(self.percents >= GlobalParameters.positive_threshold)[0]
            # Returns indicies where data is negative
            negative = np.where(self.percents <= GlobalParameters.negative_threshold)[0]
            
            # Get data to match the ratio given by the variable "percent_unmethylated"
            sample_count = int( (len(negative)/GlobalParameters.percent_unmethylated)-len(negative) )

            positive = np.random.choice(positive, size=sample_count)
            
            indexes = np.concatenate([positive, negative, np.where((GlobalParameters.positive_threshold >= self.percents) & (self.percents >= GlobalParameters.negative_threshold))[0]])

            self.percents = self.percents[indexes]
            self.sequences = self.sequences[indexes]

        

        self.percents = torch.tensor(self.percents)
        self.sequences = torch.tensor(self.sequences)

    
    def __len__(self):
        return len(self.percents)

    
    def encode(self, seq):
        # Encode sequence to token array
        encoded = [token_dict['[CLS]']]
    
        for char in seq:
            try:
                encoded += [token_dict[char.upper()]]
            except KeyError:
                encoded += [token_dict['OTHER']]
        
        encoded += [token_dict['[SEP]']]
    
        # One-hot
        encoded = F.one_hot(torch.tensor(encoded), num_classes=len(token_dict)).type(torch.float16)
    
        return encoded
        
        
    def __getitem__(self, idx):
        return self.percents[idx].to(GlobalParameters.device), self.sequences[idx].to(GlobalParameters.device)
        
def getDataLoader(data_file, training):
    
    data = MethylationDataset(data_file, training=training) 

    return DataLoader(data, batch_size=GlobalParameters.batch_size, shuffle=True, drop_last=False), len(data)