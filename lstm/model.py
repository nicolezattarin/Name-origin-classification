import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

class lstm_classifier (nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, nclasses):
        super(lstm_classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=True, 
                            dropout=0.2)

        self.linear = nn.Linear(hidden_dim, nclasses)
        self.hidden = self.init_hidden()
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # print ('embeds shape = ', embeds.shape)

        lstm_out, self.hidden = self.lstm(embeds)

        # print ('lstm_out shape = ', lstm_out.shape)
        # print ('lstm_out view shape = ', lstm_out.view(len(sentence), -1).shape)

        logits = self.linear(lstm_out)
        all_probs = self.softmax(logits)
        
        # print ('all_probs check = ', all_probs.sum(dim=1))
        # multiply element-wise the probabilities of each ngram
        # to get the probability of the name being in a class
        # print ('all_probs shape = ', all_probs.shape)
        probs = torch.prod(all_probs, dim=1)
        # print ('probslen = ', probs.shape)
        
        return probs


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






