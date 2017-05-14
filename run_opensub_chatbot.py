import tensorflow as tf
import numpy as np

# preprocessed data
from datasets.opensubtitle import data
import data_utils

metadata, idx_q, idx_a = data.load_data(PATH='datasets/opensubtitle/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

import seq2seq_wrapper

import importlib
importlib.reload(seq2seq_wrapper)

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/opensubtitle/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )

sess = model.restore_last_session()

while True:
    query = input('Input:\t')

    if query == 'quit' or query == 'exit':
        exit(0)
        
    ids = data_utils.encode(sequence=query, lookup=metadata['w2idx'])
    output = model.predict(sess, ids)
    reply = data_utils.decode(sequence=output[0], lookup=metadata['idx2w'], separator=' ')
    print('Output:'+reply)
