"""
################################################################################
                            DESCRIPTION

Functions to generate and load the vocabulary that will be used.

################################################################################
"""
from __future__ import print_function
import collections
import os
import glob

from support import tokenization, Timer
from file_support import file2str, list2file, file2list


# ==============================================================================
#                                                         VOCAB_TALLY_FROM_FILES
# ==============================================================================
def vocab_tally_from_files(data_dir):
    """ Given the root directory containing the data. It returns a
        dictionary-like of the counts of each token in the entire
        data.
    """
    ext = "txt"
    datasets = ["train", "test"]
    tally = collections.Counter([])
    for dataset in datasets:
        timer = Timer()
        for sentiment in ["neg", "pos"]:
            timer.start()
            print("Processing {} {} data".format(dataset, sentiment), end="")
            
            dir = os.path.join(data_dir, dataset, sentiment)
            files = glob.glob(os.path.join(dir, "*.{}".format(ext)))
            
            # ITERATE THROUGH EACH FILE IN THE SUBDIR - and update the tally
            for file in files:
                text = file2str(file)
                text = tokenization(text)
                tally.update(text)
                
            print("-- DONE in {}".format(timer.elapsed_string()))
    return tally


# ==============================================================================
#                                                                   CREATE_VOCAB
# ==============================================================================
def create_vocab(data_dir, n=10000):
    """ Given the directory of where the data is located, and a vocab size, n,
        It looks at all the words in the data, and trims it down to just the
        most frequent n tokens.
        
        Returns a tuple:
            - id2word:  (list) that maps id values to tokens.
                        The 0th index is reserved for 'UNKNOWN'
            - word2id:  (dict) that maps token strings to integer ids
    """
    tally = vocab_tally_from_files(data_dir=data_dir)
    tally = tally.most_common(n-1)
    id2word = ["UNKNOWN"]
    for word, _ in tally:
        id2word.append(word)
    word2id = {word: id for id, word in enumerate(id2word)}
    
    return id2word, word2id

