import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def train(model, epochs, train_loader, val_loader, optimizer, criterion, device):
    model.train()

    for epoch in range(epochs):
        print (f"Epoch {epoch}")
        print ("*"*30)
        #train
        train_loss = []
        val_loss = []
        for d in tqdm(train_loader):
            data, target = d
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            # print (output.shape)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        for d in tqdm(val_loader):
            data, target = d
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss.append(loss.item())

            
        print (f"Epoch {epoch} Train Loss: {np.mean(train_loss)} Val Loss: {np.mean(val_loss)}")


import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch LSTM')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--device', type=str, default='cpu', metavar='D',
                    help='device to use (default: cpu)')
parser.add_argument('--embedding_dim', type=int, default=200, metavar='D',
                    help='embedding dimension (default: 200)')
parser.add_argument('--hidden_dim', type=int, default=100, metavar='D',
                    help='hidden dimension (default: 100)')

#  embedding_dim, hidden_dim, vocab_size, nclasses):

def main(
    epochs=10,
    batch_size=64,
    lr=0.001,
    device='cpu',
    embedding_dim=200,
    hidden_dim=100
    ):
    from dataset import ngram_dataset
    from model import lstm_classifier

    # load data
    train_dataset = ngram_dataset(main_path='data/', train=True)
    val_dataset = ngram_dataset(main_path='data/', train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print (f"Train size: {len(train_dataset)}")

    # model
    model = lstm_classifier(
                        embedding_dim, 
                        hidden_dim,
                        train_dataset.vocab_size, 
                        train_dataset.nclasses
                    )
    model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # loss
    criterion = nn.NLLLoss()

    train(model, epochs, train_loader, val_loader, optimizer, criterion, device)

if __name__ == '__main__':
    args = parser.parse_args()
    main(
        **vars(args)
        )



