################################################################################
#                                                             IMPORT AND GLOBALS
################################################################################
from __future__ import print_function, division, unicode_literals
import glob
import os
import numpy as np

from vocab import get_vocab
from file_support import file2str
from support import str2tensor, idtensor2str
from support import Timer, pretty_time

DATA_DIR = "aclImdb"
VOCAB_FILE = "vocab.txt"
MAX_VOCAB_SIZE = 10000


################################################################################
#                                                           SUPPORTING FUNCTIONS
################################################################################

def load_data(data_dir, datasets=["train", "test"], classes=["neg", "pos"]):
    # TODO: Create validation data from the test data
    ext = "txt"  # file extensions to look for
    data = {"train": [], "test": []}
    
    # ITERATE THROUGH EACH OF THE DATASETS
    for dataset in datasets:
        timer = Timer()
        
        # ITERATE THROUGH EACH CLASS LABEL
        for class_id, sentiment in enumerate(classes):
            print("Processing {} {} ({}) data".format(dataset, sentiment,
                                                      class_id), end="")
            timer.start()
            
            # MAKE LIST OF FILES - for current subdirectory
            dir = os.path.join(DATA_DIR, dataset, sentiment)
            files = glob.glob(os.path.join(dir, "*.{}".format(ext)))
            
            # ITERATE THROUGH EACH FILE
            for file in files:
                # create tuple of [class, text from file]
                text = file2str(file)
                text = str2tensor(text, word2id=word2id)
                data[dataset].append([text, class_id])
            
            print("-- DONE in {}".format(timer.elapsed_string()))
        
        # RANDOMIZE THE ORDER OF THE TUPLES
        np.random.shuffle(data[dataset])
    
    return data


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



################################################################################
#                                                                           DATA
################################################################################
# LOAD VOCAB
id2word, word2id = get_vocab(VOCAB_FILE, DATA_DIR, MAX_VOCAB_SIZE)
n_words = len(id2word)
