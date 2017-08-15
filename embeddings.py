import numpy as np
class ReviewsGenerator(object):
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


def initialize_embeddings(n_words, embed_size):
    # INITIALIZE EMBEDDINGS TO RANDOM VALUES
    # TODO: Maybe play around with differeent initialization strategies
    init_sd = 1 / np.sqrt(embed_size)
    weights = np.random.normal(0, scale=init_sd, size=[n_words, embed_size])
    weights = weights.astype(np.float32)
    
    return weights


