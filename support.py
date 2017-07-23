from __future__ import print_function, division, unicode_literals
import os
import glob
import numpy as np
import torch
from torch.autograd import Variable
import time
from file_support import maybe_make_pardir, file2dict, dict2file


# ==============================================================================
#                                                                    PRETTY_TIME
# ==============================================================================
def pretty_time(t):
    """ Given an elapsed time in seconds, it returns the time as a string
        formatted as: "HH:MM:SS"
    """
    hours = int(t // 3600)
    mins = int((t % 3600) // 60)
    secs = int((t % 60) // 1)
    return "{:02d}:{:02d}:{:02d}".format(hours, mins, secs)


# ==============================================================================
#                                                                          TIMER
# ==============================================================================
class Timer(object):
    def __init__(self, start=True):
        """ Creates a convenient stopwatch-like timer.
            By default it starts the timer automatically as soon as it is
            created. Set start=False if you do not want this.
        """
        self.start_time = 0
        if start:
            self.start()
    
    def start(self):
        """ Start the timer """
        self.start_time = time.time()
    
    def elapsed(self):
        """ Return the number of seconds since the timer was started. """
        now = time.time()
        return (now - self.start_time)
    
    def elapsed_string(self):
        """ Return the amount of elapsed time since the timer was started as a
            formatted string in format:  "HH:MM:SS"
        """
        return pretty_time(self.elapsed())


# ==============================================================================
#                                                                   TOKENIZATION
# ==============================================================================
def tokenization(s):
    """ A Naive function for tokenization of an input string
        returns a list of the separae token strings.
    """
    s = s.lower()
    s = s.replace(".", " . ")
    s = s.replace(",", " , ")
    s = s.replace("\"", " \" ")
    s = s.replace("'", " ' ")
    s = s.replace("%", " % ")
    s = s.replace("/", " / ")
    s = s.replace("<br  / >", " ")
    s = s.replace("(", " ( ")
    s = s.replace(")", " ) ")
    s = s.replace(":", " : ")
    s = s.replace("$", " $ ")
    s = s.replace("?", " ? ")
    s = s.replace("!", " ! ")
    s = s.split()
    return s


# ==============================================================================
#                                                                     TOKENS2IDS
# ==============================================================================
def tokens2ids(tokens, word2id, unknown_id=0):
    """ Maps a list of token srings to a list of token ids.
    
    Args:
        tokens:     (list of strings) The list of token strings
        word2id:    (dict) Dictionary that maps token strings to ids
        unknown_id: (int) The id for unknown words (words not in the vocab)

    Returns: (list of ints)
    """
    return [word2id.get(token, unknown_id) for token in tokens]


# ==============================================================================
#                                                                        STR2IDS
# ==============================================================================
def str2ids(s, word2id, unknown_id=1):
    """ Given a string s, and a dictionary that maps from  tokens
        to an index representing that word, it returns the string
        represented as a list of token ids.
    """
    line = tokenization(s)
    line = tokens2ids(line, word2id=word2id, unknown_id=unknown_id)
    return line


# ==============================================================================
#                                                                     STR2TENSOR
# ==============================================================================
def str2tensor(s, word2id, unknown_id=0):
    """ Given a string, and a dictionary that maps each word to an
        integer representing the embedding index, it converts the sequence
        of characters in s to a pytorch Variable tensor of character ids.
    """
    ids = str2ids(s, word2id, unknown_id=unknown_id)
    return (Variable(torch.LongTensor(ids)))


# ==============================================================================
#                                                                   IDTENSOR2STR
# ==============================================================================
def idtensor2str(t, id2word):
    """ Given a tensor of token ids, it returns a human readable string"""
    return " ".join([id2word[id] for id in t.data.numpy()])


# ==============================================================================
#                                                                        IDS2STR
# ==============================================================================
def ids2str(a, id2word):
    """ Given a list of token ids, it returns a human readable string"""
    return " ".join([id2word[id] for id in a])


# ==============================================================================
#                                                         PROCESS_LINE_FOR_BATCH
# ==============================================================================
def process_line_for_batch(a, maxlen, padval=0):
    """ Given a list of items, it returns a version of that list
        with the length limited to  `maxlen`.  Lists  with  more
        elements than  `maxlen`  will  be trimmed, and any lists
        shorter than this will be padded with  `padval`  at  the
        start.
        
    NOTE:
        Currently, the trimming that is performed, is that it takes
        the first `maxlen` items in the list.
    """
    # TODO: Select a random subsection instead of just fist maxlen items
    
    if len(a) > maxlen:
        a = a[:maxlen]
    elif len(a) < maxlen:
        a = np.pad(a, (maxlen - len(a), 0), 'constant', constant_values=padval)
    return a


# ==============================================================================
#                                                             BATCH_FROM_INDICES
# ==============================================================================
def batch_from_indices(x, y=None, ids=[0], maxlen=100, padval=0):
    """ Given the input sequences x (and optionally output labels y),
        and a list of indices to use for the batch, it  extracts  the
        sequences  at  those  indices,  keeping  each  sequence  to a
        maximum  length  of `maxlen`. Any sequences  longer than this
        will be trimmed, and any sequences shorter than this will  be
        padded with `padval` at the start.

    Args:
        x:          (array of array of ints)
                    The input sequences
        y:          (array of ints or None)(optional)
                    The output labels
        ids:        (list of ints)
                    The indices to extract.
        maxlen:     (int)(default=100)
                    Clip sequences to be no longer than this length,
                    and pad anything shorter.
        padval:     (int)(default=0)
                    Value to use for padding.

    Returns:
        If `y` is None, then it just returns xbatch, otherwise it
        returns a tuple (xbatch, ybatch)
    """
    # INITIALIZE EMPTY BATCH OF ARRAYS
    batchsize = len(ids)
    xbatch = np.empty((batchsize, maxlen), dtype=np.int64)
    if y:
        ybatch = np.empty(batchsize, dtype=np.int64)
    
    # EXTRACT ITEMS FROM DATA - clipping or padding lengths to maxlen
    for i, idx in enumerate(ids):
        xbatch[i] = process_line_for_batch(x[idx], maxlen=maxlen, padval=padval)
        if y:
            ybatch[i] = y[idx]
    
    # CONVERT TO PYTORCH VARIABLES
    xbatch = Variable(torch.LongTensor(xbatch))
    if y:
        ybatch = Variable(torch.LongTensor(ybatch))
    
    if y:
        return xbatch, ybatch
    else:
        return xbatch


# ==============================================================================
#                                                            CREATE_RANDOM_BATCH
# ==============================================================================
def create_random_batch(x, y=None, batchsize=32, maxlen=100, padval=0):
    """ Given the input sequences x (and optionally output labels y),
        it  randomly samples a `batchsize` sized batch, keeping each
        sequence  to  a  maximum  length  of `maxlen`. Any sequences
        longer than  this will be trimmed, and any sequences shorter
        than this will  be padded with `padval` at the start.

    Args:
        x:          (array of array of ints)
                    The input sequences
        y:          (array of ints or None)(optional)
                    The output labels
        batchsize:  (int)(default=32)
                    How many samples to use in the batch.
        maxlen:     (int)(default=100)
                    Clip sequences to be no longer than this length,
                    and pad anything shorter.
        padval:     (int)(default=0)
                    Value to use for padding.

    Returns:
        If `y` is None, then it just returns xbatch, otherwise it
        returns a tuple (xbatch, ybatch)
    """
    ids = np.random.randint(0, len(x), size=batchsize, dtype=np.int64)
    return batch_from_indices(x, y, ids=ids, maxlen=maxlen, padval=padval)
    

# ==============================================================================
#                                                                  TAKE_SNAPSHOT
# ==============================================================================
def take_snapshot(model, file, verbose=True):
    """ Takes a snapshot of all the parameter values of a model.

    Args:
        model: (Model Object)
        file:  (str) filepath to save file as
        verbose: (bool)(default=True) whether it should print out feedback.
    """
    maybe_make_pardir(file)
    torch.save(model.state_dict(), file)
    if verbose:
        print("SAVED SNAPSHOT: {}".format(file))


# ==============================================================================
#                                                                  LOAD_SNAPSHOT
# ==============================================================================
def load_snapshot(model, file):
    """ Given a model, and the path to a snapshot file, It loads the
        parameters from that snapshot file.
    """
    model.load_state_dict(torch.load(file))


# ==============================================================================
#                                                                 EPOCH_SNAPSHOT
# ==============================================================================
def epoch_snapshot(model, epoch, loss, name, dir, verbose=True):
    """ Takes a snapshot of all the parameter values of a model.

    Args:
        model: (Model Object)
        epoch: (int)
        loss:  (float)
        name:  (str) model name
        dir:   (str) directory where snapshots will be taken
        verbose: (bool)(default=True) whether it should print out feedback.
    """
    template = "{model}_{epoch:05d}_{loss:06.3f}.params"
    filename = template.format(model=name, epoch=epoch, loss=loss)
    filepath = os.path.join(dir, filename)
    
    take_snapshot(model, filepath, verbose=verbose)


# ==============================================================================
#                                                           LOAD_LATEST_SNAPSHOT
# ==============================================================================
def load_latest_snapshot(model, dir):
    """ Given a model, and the path to the dir containing the snapshots,
        It loads the parameters from the latest saved snapshot.

        If file, does not exits, then it does nothing.
    """
    try:
        params_file = sorted(glob.glob(os.path.join(dir, "*.params")))[-1]
        model.load_state_dict(torch.load(params_file))
        print("LOADING PARAMETERS FROM:", params_file)
    except IndexError:
        print("USING MODELS INITIAL PARAMETERS")


# ==============================================================================
#                                                              LOAD_HYPER_PARAMS
# ==============================================================================
def load_hyper_params(file):
    """ Given a text file containing the models hyper-parameters, it returns
        a dictionary. of those items.

        The text file should be in the following format:

            MAX_VOCAB: 10000
            SAMPLE_LENGTH: 100
            BATCH_SIZE: 128
            N_HIDDEN: 128
            EMBED_SIZE: 64
            N_LAYERS: 1
            DROPOUT: 0.3
            ALPHA: 0.01
            
        Any key: value pairs that are not included in the file will be
        replaced with the default values shown in the above example.

        An additional optional key value pair may be included.

            LAST_ALPHA: 0.01

        This represents the last alpha that was used by the model.
        If this key value pair is not included in the file, then
        it will be created, using the same value from ALPHA.
    """
    # If file exists load settings from file.
    # Otherwise, create default dictionary
    if os.path.exists(file) and os.path.isfile(file):
        d = file2dict(file)
    else:
        d = {}
    
    # Use defaults for missing items
    d.setdefault("MAX_VOCAB", 10000)
    d.setdefault("SAMPLE_LENGTH", 100)
    d.setdefault("BATCH_SIZE", 128)
    d.setdefault("N_HIDDEN", 128)
    d.setdefault("EMBED_SIZE", 64)
    d.setdefault("N_LAYERS", 1)
    d.setdefault("DROPOUT", 0.3)
    d.setdefault("ALPHA", 0.01)
    
    # Convert to correct data types
    d["MAX_VOCAB"] = int(d["MAX_VOCAB"])
    d["SAMPLE_LENGTH"] = int(d["SAMPLE_LENGTH"])
    d["BATCH_SIZE"] = int(d["BATCH_SIZE"])
    d["N_HIDDEN"] = int(d["N_HIDDEN"])
    d["EMBED_SIZE"] = int(d["EMBED_SIZE"])
    d["N_LAYERS"] = int(d["N_LAYERS"])
    d["DROPOUT"] = float(d["DROPOUT"])
    d["ALPHA"] = float(d["ALPHA"])
    
    # Latest alpha
    d.setdefault("LAST_ALPHA", d["ALPHA"])
    d["LAST_ALPHA"] = float(d["LAST_ALPHA"])
    
    return d


# ==============================================================================
#                                                              SAVE_HYPER_PARAMS
# ==============================================================================
def save_hyper_params(d, file):
    """ Given dictionary containing the hyperparameter settings,
        and and file path to save to, it saves the dictionary
        contents as a text file, in the following format:

            SAMPLE_LENGTH: 200
            BATCH_SIZE: 32
            N_HIDDEN: 128
            EMBED_SIZE: 128
            N_LAYERS: 1
            DROPOUT: 0.7
            ALPHA: 0.01
            LAST_ALPHA: 0.01
    """
    order = ["MAX_VOCAB",
             "SAMPLE_LENGTH",
             "BATCH_SIZE",
             "N_HIDDEN",
             "EMBED_SIZE",
             "N_LAYERS",
             "DROPOUT",
             "ALPHA",
             "LAST_ALPHA"]
    dict2file(d, file, keys=order)


