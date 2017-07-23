from support import batch_from_indices
import numpy as np

# ==============================================================================
#                                                              BATCH_PREDICTIONS
# ==============================================================================
def batch_predictions(model, x, seq_maxlen=100, padval=0):
    """ Given a model and batch of data, it returns the predictions
    
    Args:
        model:       (pytorch model)
        x:           (list of list of ints)
                     The input data, consisting of multiple sequences,
                     where each sequence is a list of token ids.
        seq_maxlen:  (int)(default=100)
                     Trim/pad sequences to this length.
        padval:      (int) (default=0)
                     Value to use for padding.

    Returns: (numpy array)
        The array of predicted class ids for each sample in the batch.
    """
    n_batch = len(x)
    
    # Set to evaluation mode, and store original training state
    train_state = model.training
    model.eval()
    
    # Initialize model
    hidden = model.init_hidden(batch_size=n_batch)
    model.zero_grad()
    
    # prepare data
    X = batch_from_indices(x, ids=range(n_batch), maxlen=maxlen, padval=padval)
    out, _ = model(X, hidden)
    _, index = out.data.topk(1)
    
    # return model back to original training state
    model.train(train_state)
    
    return index.view(-1).numpy()


