from __future__ import print_function, division, unicode_literals
import os
import glob
import numpy as np

from file_support import file2str, pickle2obj, obj2pickle
from file_support import file2list
from support import str2ids, Timer


# ==============================================================================
#                                                                      LOAD_DATA
# ==============================================================================
def load_data(data_dir, vocab_file, classes=["neg", "pos"], valid_ratio=0.3, seed=0):
    """ Given the root directory containing the IMDB data. It returns a
        dictionary with separate keys for the train and test datasets.
        {"xtrain":[...]
         "ytrain": [...]
         "xtest": [...]
         "ytest": [...]

         }
         
        Each value is a list of lists, where each inner list contains the
        sequence of token ids for that review.

    Args:
        data_dir: (str) Path to the root directory containing the data
        vocab_file:  (str) File containing the vocab
        classes:

    Returns:
        (dict)
    """
    id2word = file2list(vocab_file)
    word2id = {word: id for id, word in enumerate(id2word)}

    ext = "txt"  # file extensions to look for
    data = {"xtrain": [],
            "ytrain": [],
            "xtest" : [],
            "ytest" : [],
            "xvalid": [],
            "yvalid": []}
    
    # ITERATE THROUGH EACH OF THE DATASETS
    datasets = ["train", "test"]
    for dataset in datasets:
        timer = Timer()
        
        # ITERATE THROUGH EACH CLASS LABEL
        for class_id, class_name in enumerate(classes):
            print("Processing {} {} ({}) data".format(dataset, class_name,
                                                      class_id), end="")
            timer.start()
            
            # MAKE LIST OF FILES - for current subdirectory
            dir = os.path.join(data_dir, dataset, class_name)
            files = glob.glob(os.path.join(dir, "*.{}".format(ext)))
            
            # ITERATE THROUGH EACH FILE
            for file in files:
                # Create input features and labels
                text = file2str(file)
                text = str2ids(text, word2id=word2id, unknown_id=1)
                data["x" + dataset].append(text)
                data["y" + dataset].append(class_id)
            
            print("-- DONE in {}".format(timer.elapsed_string()))
        
        # RANDOMIZE THE ORDER OF THE data
        # TODO: Consider using a different method that does it in place
        n = len(data["y" + dataset])
        np.random.seed(seed=seed)
        ids = np.random.permutation(n)
        data["x" + dataset] = map(lambda id: data["x" + dataset][id], ids)
        data["y" + dataset] = map(lambda id: data["y" + dataset][id], ids)
        
        # VALIDATION DATA - Split a portion of train data for validation
        n_valid = int(len(data["ytrain"]) * valid_ratio)
        data["xvalid"] = data["xtrain"][:n_valid]
        data["yvalid"] = data["ytrain"][:n_valid]
        data["xtrain"] = data["xtrain"][n_valid:]
        data["ytrain"] = data["ytrain"][n_valid:]
    
    return data


# ==============================================================================
#                                                                       GET_DATA
# ==============================================================================
def get_data(data_dir, cached_data, vocab_file):
    """ Loads cached data (as sequences of word ids) if it exists, otherwise it
        creates the dataset from the raw IMDB text files and caches the
        processed dataset.
    
    Args:
        data_dir:       (str) The IMDB root directory containing the "train"
                        and "test" subdirectories.
        cached_data:    (str) The path to the pickle file contianing the
                        cached data
        vocab_file:     (str) The file that contains the vocabulary, one
                        token per line in order from most frequent to
                        least frequent.

    Returns:
        (dict)
    """
    if os.path.exists(cached_data):
        print("LOADING CACHED DATA")
        data = pickle2obj(cached_data)
    else:
        print("PROCESSING RAW DATA")
        data = load_data(data_dir=data_dir, vocab_file=vocab_file,
                         valid_ratio=0.3, seed=45)
        print("CACHING DATA")
        obj2pickle(data, cached_data)

    return data


