################################################################################
#                                                             IMPORT AND GLOBALS
################################################################################
from __future__ import print_function, division, unicode_literals
import glob
import os
import numpy as np

from vocab import get_vocab
from file_support import file2str
from support import str2ids, idtensor2str
from support import Timer, pretty_time

DATA_DIR = "aclImdb"
from model import Model

VOCAB_FILE = "vocab.txt"
MAX_VOCAB_SIZE = 10000


################################################################################
#                                                           SUPPORTING FUNCTIONS
################################################################################

# ==============================================================================
#                                                                      LOAD_DATA
# ==============================================================================
def load_data(data_dir, datasets=["train", "test"], classes=["neg", "pos"]):
    # TODO: Create validation data from the test data
    ext = "txt"  # file extensions to look for
    data = {"xtrain": [],
            "ytrain": [],
            "xtest": [],
            "ytest": []}
    
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
                text = str2ids(text, word2id=word2id)
                data["x"+dataset].append(text)
                data["y"+dataset].append(class_id)
            
            print("-- DONE in {}".format(timer.elapsed_string()))
        
        # RANDOMIZE THE ORDER OF THE data
        # TODO: Consider using a different method that does it in place
        n = len(data["y"+dataset])
        ids = np.random.permutation(n)
        data["x" + dataset] = map(lambda id: data["x" + dataset][id], ids)
        data["y" + dataset] = map(lambda id: data["y" + dataset][id], ids)
        
    return data


# ==============================================================================
#                                                                     TRAIN_STEP
# ==============================================================================
def train_step(model, X, Y):
    """ Given a model, the input X representing a batch of  sequences  of
        words, and the output labels Y, it performs a full training step,
        updating the model parameters. based on the loss and optimizer of
        the model.

    Args:
        model:      (Model object)
                    The model containing the neural net architecture.
        X:          (torch Variable)
                    The input tensor of shape: [sequence_length]
        Y:          (torch Variable)
                    The output labels tensor of shape: [1]

    Returns:
        Returns the average loss over the batch of sequences.
    """
    # Get dimensions of input and target labels
    batch_size, sample_length = X.size()
    
    # Initialize hidden state and reset the accumulated gradients
    hidden = model.init_hidden(batch_size)
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
# TODO: make vocab files named "vocab_10000.txt" where the number is vocab size
id2word, word2id = get_vocab(VOCAB_FILE, DATA_DIR, MAX_VOCAB_SIZE)
n_words = len(id2word)

# CLASS MAPPINGS
id2class = ["neg", "pos"]
class2id = {label:id for id, label in enumerate(id2class)}

# LOAD DATA
data = load_data(data_dir=DATA_DIR, datasets=["train", "test"])


################################################################################
#                                                                          MODEL
################################################################################
model = Model(n_vocab=n_words, embed_size=64, h_size=128, n_layers=1, dropout=0.3)
model.update_learning_rate(0.001)

