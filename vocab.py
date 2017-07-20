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


