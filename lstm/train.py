import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

def train(model, epochs, train_loader, val_loader, optimizer, criterion, device, save_path):
    # load model if exists
    if os.path.exists(os.path.join(save_path, f"model.pth")):
        model.load_state_dict(torch.load(os.path.join(save_path, f"model.pth")))
        print ("model loaded")
    else:
        print ("model not found, training from scratch")
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

        # save model
        torch.save(model.state_dict(), os.path.join(save_path, f"model.pth"))
        #save loss
        np.save(os.path.join(save_path, f"train_loss.npy"), np.array(train_loss))
        np.save(os.path.join(save_path, f"val_loss.npy"), np.array(val_loss))

        print (f"Epoch {epoch} Train Loss: {np.mean(train_loss)} Val Loss: {np.mean(val_loss)}")

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
def metrics (model, test_loader, device):
    model.eval()
    all_preds = []
    correct = []
    with torch.no_grad():
        for d in tqdm(test_loader):
            data, target = d
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.append(pred.cpu().numpy())
            correct.append(target.cpu().numpy())

    acc = accuracy_score(np.concatenate(correct), np.concatenate(all_preds))
    f1  = f1_score(np.concatenate(correct), np.concatenate(all_preds), average='macro')
    prec = precision_score(np.concatenate(correct), np.concatenate(all_preds), average='macro')
    recall = recall_score(np.concatenate(correct), np.concatenate(all_preds), average='macro')

    #classificaiton report
    from sklearn.metrics import classification_report
    print (classification_report(np.concatenate(correct), np.concatenate(all_preds)))

    metric = {
        'acc': acc,
        'f1': f1,
        'prec': prec,
        'recall': recall
    }
    return metric
    
        

import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch LSTM')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--device', type=str, default='cpu', metavar='D',
                    help='device to use (default: cpu)')
parser.add_argument('--embedding_dim', type=int, default=200, metavar='D',
                    help='embedding dimension (default: 200)')
parser.add_argument('--hidden_dim', type=int, default=100, metavar='D',
                    help='hidden dimension (default: 100)')
parser.add_argument('--test', action='store_true', default=True,
                    help='test mode')
parser.add_argument('--ngram', type=int , default=2, metavar='N',
                    help='ngram (default: 2)')
parser.add_argument('--data_file', type=str, default='data', metavar='M',
                    help='data file (default: lstm)')
#  embedding_dim, hidden_dim, vocab_size, nclasses):

def main(
    epochs=10,
    batch_size=64,
    lr=0.001,
    device='cpu',
    embedding_dim=200,
    hidden_dim=100,
    test=False,
    ngram=2,
    data_file='data'
    ):
    from dataset import ngram_dataset
    from model import lstm_classifier

    # load data
    path = os.path.join(data_file, f'ngram_{ngram}')

    train_dataset = ngram_dataset(main_path=path+'/', train=True)
    val_dataset = ngram_dataset(main_path=path+'/', train=False)

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

    if not test:
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        # loss
        criterion = nn.CrossEntropyLoss()
        train(model, epochs, train_loader, val_loader, optimizer, criterion, device, save_path=path)

    else:
        model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
        test_metric = metrics(model, val_loader, device)
        train_metric = metrics(model, train_loader, device)


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        **vars(args)
        )



