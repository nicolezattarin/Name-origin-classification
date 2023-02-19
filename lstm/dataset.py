from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
# dataloader
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# dataset
class ngram_dataset(Dataset):
    def __init__(self, main_path, train=True, ndata=None):
        self.train = train
        if self.train:
            self.X = np.loadtxt(main_path + 'X_train.txt')
            self.y = np.loadtxt(main_path + 'y_train.txt', dtype=int)
        else:
            self.X = np.loadtxt(main_path + 'X_test.txt')
            self.y = np.loadtxt(main_path + 'y_test.txt', dtype=int)    
    
        if ndata is not None:
            self.X = self.X[:ndata]
            self.y = self.y[:ndata]

        self.vocab = pd.read_csv(main_path + 'vocab.txt', header=None)

        self.classes = pd.read_csv(main_path + 'idx2race.txt', header=None)
        self.classes = {i: c for i, c in enumerate(self.classes[0])}

        self.vocab_size = len(self.vocab) + 1 # unk token 
        self.nclasses = len(self.classes)
        print (f"vocab_size = {self.vocab_size}")
        print (f"nclasses = {self.nclasses}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return (x, y)
       
