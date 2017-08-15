import numpy as np
def initialize_embeddings(n_words, embed_size):
    # INITIALIZE EMBEDDINGS TO RANDOM VALUES
    # TODO: Maybe play around with differeent initialization strategies
    init_sd = 1 / np.sqrt(embed_size)
    weights = np.random.normal(0, scale=init_sd, size=[n_words, embed_size])
    weights = weights.astype(np.float32)
    
    return weights



