import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

class lstm_classifier (nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, nclasses, nlayers=1, dropout=0.4):
        super(lstm_classifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=True, 
                            num_layers=nlayers,
                            dropout=dropout,
                            bidirectional=True
                            )

        self.linear = nn.Linear(hidden_dim, nclasses)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # print ('embeds shape = ', embeds.shape)

        lstm_out,  (ht, ct) = self.lstm(embeds)

        logits = self.linear(ht[-1])
        return logits


if __name__ == '__main__':
    
    data_path = 'data/X_train.txt'
    X_train = np.loadtxt(data_path, dtype=int)
    vocab = np.loadtxt('data/vocab.txt', dtype=str)
    classes = np.loadtxt('data/idx2race.txt', dtype=str)

    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 100
    nclasses = len(classes)
    print (f"vocab_size = {vocab_size}")
    print (f"nclasses = {nclasses}")


    # convert to torch tensor
    X = torch.tensor(X_train[0], dtype=torch.long).unsqueeze(0)

    # create model
    # add a batch dimension
    print 
    print (f"sample shape = {X.shape}")
    model = lstm_classifier(embedding_dim, hidden_dim, vocab_size, nclasses)

    # test a forward pass
    # logits for each input ngram int the name 
    logits = model(X)

    # we want the final probability to be in a single class
    # so for each class we take the logit -> apply softmax -> take the product among all the ngrams
    # this is the probability of the name being in that class

    print(logits.shape)






