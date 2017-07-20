################################################################################
#                                                             IMPORT AND GLOBALS
################################################################################
from __future__ import print_function, division, unicode_literals
import glob
import os
import numpy as np

from vocab import get_vocab
from file_support import file2str
from support import Timer, pretty_time

DATA_DIR = "aclImdb"
VOCAB_FILE = "vocab.txt"
MAX_VOCAB_SIZE = 10000


def train(model, X, Y):
    # Get dimensions of input and target labels
    msg = "X and Y should be only one axis in shape"
    assert len(X.size()) == len(Y.size()) == 1, msg
    batch_size=1
    sample_length = X.size()[0]
    
    # Initialize hidden state and reset the accumulated gradients
    hidden = model.init_hidden()
    model.zero_grad()
    
    # Run through the sequence on the LSTM
    output, _ = model(X, hidden)
    
    # Calculate gradients, and update parameter weights
    loss = model.loss_func(output, Y)
    loss.backward()
    model.optimizer.step()
    
    # Return the loss
    return loss.data[0]

