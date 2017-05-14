PUNCTUATION_CHAR = set(['.', '..', '...', '....', '.....', '......', ',', ',,', ',,,', '!', '!!', '!!!', '?', '??', '???', '????'])
PUNCTUATION_IDS = set([])

limit = {
        'maxq' : 20,
        'minq' : 0,
        'maxa' : 20,
        'mina' : 3
        }

UNK = 'UNknown'
VOCAB_SIZE = 25000

import random
import sys

import nltk
import itertools
from collections import defaultdict

import numpy as np

import pickle

def process_data(raw_file, vocab_file):
    idx2w = ['_']  # 0 is invalid value for NN
    w2idx = {}
    freq_dist = nltk.FreqDist()
    with open(vocab_file, 'r') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            idx2w.append(word)
            w2idx[word] = idx
            freq_dist[word] = 0
            if word in PUNCTUATION_CHAR:
                PUNCTUATION_IDS.add(idx)

    idx_q = []
    idx_a = []
    with open(raw_file, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('|')
            q_ids = [int(val) for val in parts[0].split(' ')]
            a_ids = [int(val) for val in parts[1].split(' ')]
            for val in q_ids:
                word = idx2w[val]
                freq_dist[word] += 1
            for val in a_ids:
                word = idx2w[val]
                freq_dist[word] += 1

            #Filter out some long sentences
            if len(q_ids) > limit['maxq'] or len(a_ids) > limit['maxa']:
                continue

            #Cancel the last punctuation
            if q_ids[-1] in PUNCTUATION_IDS:
                q_ids[-1] = 0
            if a_ids[-1] in PUNCTUATION_IDS:
                a_ids[-1] = 0
            #Padding
            q_ids += [0] * (limit['maxq'] - len(q_ids))
            a_ids += [0] * (limit['maxa'] - len(a_ids))
            idx_q.append(q_ids)
            idx_a.append(a_ids)

    # Source and Target Set: (DataSize, 20)
    idx_q = np.array(idx_q)
    idx_a = np.array(idx_a)

    # save them
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a


if __name__ == '__main__':
    process_data('s_given_t_train.txt', 'movie_25000')
