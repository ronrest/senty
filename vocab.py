"""
################################################################################
                            DESCRIPTION

Functions to generate and load the vocabulary that will be used.


PROBLEMATIC FILES - The following files are known to have encoding issues, and
are skipped.
    - "train/unsup/2759_0.txt"

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
        for subdir in ["neg", "pos", "unsup"]:
            timer.start()
            print("Processing {} {} data".format(dataset, subdir), end="")
            
            dir = os.path.join(data_dir, dataset, subdir)
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
def create_vocab(data_dir, n=None):
    """ Given the directory of where the data is located, and a vocab size, n,
        It looks at all the words in the data, and trims it down to just the
        most frequent n tokens.
        
        If n is None (default) it returns all the words, in the order from most
        common to least common.
        
        Returns a tuple:
            - id2word:  (list) that maps id values to tokens.
                        The 0th index is reserved for 'PAD',
                        The 1st index is reserved for 'UNKNOWN'
            - word2id:  (dict) that maps token strings to integer ids
    """
    tally = vocab_tally_from_files(data_dir=data_dir)
    
    # Limit vocab size
    if n is not None:
        assert n > 2, "n must be greater than 2!"
        tally = tally.most_common(n-2)
    else:
        tally = tally.most_common()
    
    # Add padding and unknown tokens
    id2word = ["PAD", "UNKNOWN"]
    for word, _ in tally:
        id2word.append(word)
    
    word2id = {word: id for id, word in enumerate(id2word)}
    return id2word, word2id


# ==============================================================================
#                                                                      GET_VOCAB
# ==============================================================================
def get_vocab(vocab_file, data_dir, max_vocab_size=None):
    """ Gets the vocabulary.
        
        If `vocab_file` has already been created, then it loads the vocabulary
        from there, otherwise, it generates the vocabulary by looking at the
        raw data, and caches the vocabulary in `vocab_file`.
        
        You can optionally set the max size of the vocabulary. By default
        it is set to None, which means it uses every sinmgle token in the
        data.
        
    Returns a tuple:
            - id2word:  (list) that maps id values to tokens.
                        The 0th index is reserved for 'UNKNOWN'
            - word2id:  (dict) that maps token strings to integer ids
    """
    # GET VOCABULARY
    if not os.path.exists(vocab_file):
        print("GENERATING VOCAB FROM RAW DATA")
        id2word, word2id = create_vocab(data_dir, n=None)
        
        # Cache vocab to file
        print("Caching vocab to ", vocab_file)
        list2file(id2word, file=vocab_file)
    
    print("LOADING VOCAB FROM PRE-CACHED FILE")
    id2word = file2list(vocab_file)[:max_vocab_size]
    word2id = {word: id for id, word in enumerate(id2word)}

    # n_vocab = len(id2word)
    return id2word, word2id

