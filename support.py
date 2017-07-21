from __future__ import print_function, division, unicode_literals
import torch
from torch.autograd import Variable
import time


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
def str2ids(s, word2id, unknown_id=0):
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

def process_line_for_batch(a, maxlen, padval=0):
    # TODO: Select a random subsection instead of just fist maxlen items
    
    if len(a) > maxlen:
        a = a[:maxlen]
    elif len(a) < maxlen:
        a = np.pad(a, (maxlen - len(a), 0), 'constant', constant_values=padval)
    return a


