import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ngrams', type=int, default=2)
parser.add_argument('--filepath', type=str, default='data/wiki_name_race.txt')
parser.add_argument('--save_path', type=str, default='data')

def main (
    ngrams,
    filepath,
    save_path
    ):

    print ("ngrams = %d" % ngrams)
    print ("filepath = %s" % filepath)

    print ("Loading data...")
    # Wikilabels
    df = pd.read_csv(filepath)
    df.dropna(subset=['name_first', 'name_last'], inplace=True)
    # add middle name to first name
    df['name_first'] = df['name_first'] + ' ' + df['name_middle'].fillna('')
    # drop middle name column
    df = df.drop('name_middle', axis=1)
    # concat last name and first name
    df['name_last_name_first'] = df['name_last'] + ' ' + df['name_first'] 

    print ("creating n-gram vocabulary...") 
    # build n-gram list
    vect = CountVectorizer(analyzer='char', ngram_range=(ngrams, ngrams), lowercase=False) 
    a = vect.fit_transform(df.name_last_name_first)
    vocab = vect.vocabulary_ 

    # sort n-gram by freq (highest -> lowest)
    words = []
    for b in vocab:
        freq = vocab[b]
        
        words.append((a[:, freq].sum(), b))
        #break
    words = sorted(words, reverse=True)
    words_list = ['<UNK>']
    words_list.extend([w[1] for w in words])
    num_words = len(words_list)
    print("num_words = %d" % num_words)

    ngram2idx = {ngram: i for i, ngram in enumerate(words_list)}
    idx2ngram = {i: ngram for i, ngram in enumerate(words_list)}

    print ("Creating dataset...") 
    # build X from index of n-gram sequence
    X = np.array(df.name_last_name_first.apply(lambda c: [ngram2idx.get(ngram, 0) for ngram in [c[i:i+ngrams] for i in range(len(c)-ngrams+1)]]))

    races = np.unique(df.race.astype('category'))
    race2idx = {x:i for i,x in enumerate(races)}
    idx2race = {i:x for i,x in enumerate(races)}

    y = np.array(df.race.apply(lambda c: race2idx[c]))

    # Split train and test dataset
    X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
    print ("X_train.shape = %s" % str(X_train.shape))
    print ("y_train.shape = %s" % str(y_train.shape))
    print ("X_test.shape = %s" % str(X_test.shape))
    print ("y_test.shape = %s" % str(y_test.shape))

    # padd sequences to the 80th percentile
    maxlen = np.percentile([len(x) for x in X_train], 80)
    maxlen = int(maxlen)
    print("maxlen = %d" % maxlen)

    # pad with 0s (i.e. UNK token)
    from keras.utils import pad_sequences
    X_train = pad_sequences(X_train, maxlen=maxlen, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=maxlen, padding='post', truncating='post')

    print ("Saving dataset...")
    # save dataset 
    np.savetxt(os.path.join(save_path, 'X_train.txt'), X_train, fmt='%d')
    np.savetxt(os.path.join(save_path, 'X_test.txt'), X_test, fmt='%d')
    np.savetxt(os.path.join(save_path, 'y_train.txt'), y_train, fmt='%d')
    np.savetxt(os.path.join(save_path, 'y_test.txt'), y_test, fmt='%d')

    # save vocab
    with open(os.path.join(save_path, 'vocab.txt'), 'w') as f:
        for v, fr in vocab.items():
            f.write(f"{v}\t{fr}\n")

    # save ngram2idx
    with open(os.path.join(save_path, 'ngram2idx.txt'), 'w') as f:
        for v, fr in ngram2idx.items():
            f.write(f"{v}\t{fr}\n")

    # save idx2ngram
    with open(os.path.join(save_path, 'idx2ngram.txt'), 'w') as f:
        for v, fr in idx2ngram.items():
            f.write(f"{v}\t{fr}\n")

    # save race2idx
    with open(os.path.join(save_path, 'race2idx.txt'), 'w') as f:
        for v, fr in race2idx.items():
            f.write(f"{v}\t{fr}\n")

    # save idx2race
    with open(os.path.join(save_path, 'idx2race.txt'), 'w') as f:
        for v, fr in idx2race.items():
            f.write(f"{v}\t{fr}\n")

    print ("Done!")

if __name__ == '__main__':
    args = parser.parse_args()
    main(
        **vars(args)
    )