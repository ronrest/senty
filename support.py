from __future__ import print_function, division, unicode_literals
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



