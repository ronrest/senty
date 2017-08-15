import os
from io import open
from support import tokenization
import numpy as np


# ==============================================================================
#                                                              REVIEWS GENERATOR
# ==============================================================================
class ReviewsGenerator(object):
    """ A python generator that iterates through every review in the imdb
        dataset, one review at a time from the 'train' subdirectory. This is
        designed to be used to train the word vectors.
    """
    def __init__(self, rootdir):
        self.rootdir = rootdir

    def __iter__(self):
        for subdir in ["pos", "neg", "unsup"]:
            dirpath = os.path.join(self.rootdir, "train", subdir)
            for filename in os.listdir(dirpath):
                filepath = os.path.join(dirpath, filename)
                with open(filepath , mode="r", encoding="utf-8") as fileobj:
                    for line in fileobj:
                        yield tokenization(line)


# ==============================================================================
#                                                          INITIALIZE_EMBEDDINGS
# ==============================================================================
def initialize_embeddings(n_words, embed_size):
    """ Creates a numpy array n_words*embed_size to be used for word
        embeddings initialized using a basic variant of Xavier
        initialization"""
    # TODO: Maybe play around with different initialization strategies
    init_sd = 1 / np.sqrt(embed_size)
    weights = np.random.normal(0, scale=init_sd, size=[n_words, embed_size])
    weights = weights.astype(np.float32)
    
    return weights


# ==============================================================================
#                                                       EXTRACT_GLOVE_EMBEDDINGS
# ==============================================================================
def extract_glove_embeddings(file, n_words, embed_size, word2id):
    """ Given a text file that contains pretrained glove embeddings
        it returns a numpy array where each row is a word vector,
        and row ids exactly match the word order from our vocab.

    Args:
        file:       (str) path to the embeddings text file
        n_words:    (int) Number of words to use for vocab
        embed_size: (int) Word vector size
        word2id:    (dict) dict that maps from words to indices

    Returns: (numpy array)
        A numpy array of shape n_words * embed_size of the word embeddings
        whose row order is based on the index mappings in `word2id`.
    """
    weights = initialize_embeddings(n_words, embed_size)
    
    # EXTRACT DESIRED GLOVE WORD VECTORS FROM TEXT FILE
    with open(file, encoding="utf-8", mode="r") as textFile:
        for line in textFile:
            # Extract the word, and embeddings vector
            line = line.split()
            word, vector = (line[0], np.array(line[1:], dtype=np.float32))
            
            # If it is in our vocab, then update the corresponding weights
            id = word2id.get(word, None)
            if id is not None:
                weights[id] = vector
    return weights



