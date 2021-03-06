import os
import numpy as np

from support import batch_from_indices
from file_support import pickle2obj


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
    X = batch_from_indices(x, ids=range(n_batch), maxlen=seq_maxlen, padval=padval)
    out, _ = model(X, hidden)
    _, index = out.data.topk(1)
    
    # return model back to original training state
    model.train(train_state)
    
    return index.view(-1).numpy()


# ==============================================================================
#                                                                    PREDICTIONS
# ==============================================================================
def predictions(model, x, batch_size=128, seq_maxlen=100, padval=0):
    """ Given a model and all the input data you want predictions for,
        it returns the predictions. It splits the data up into batches
        to prevent memory bloat.

    Args:
        model:       (pytorch model)
        x:           (list of list of ints)
                     The input data, consisting of multiple sequences,
                     where each sequence is a list of token ids.
        batch_size:  (int)(default=128)
                     How many samples to use for each batch.
        seq_maxlen:  (int)(default=100)
                     Trim/pad sequences to this length.
        padval:      (int) (default=0)
                     Value to use for padding.

    Returns: (numpy array)
        The array of predicted class ids for each sample in the batch.
    """
    n_samples = len(x)
    n_batches = int(np.ceil(n_samples / batch_size))
    preds = np.ones(shape=n_samples, dtype=np.int8) * 66
    for i in range(0, n_samples, batch_size):
        batch = x[i: i + batch_size]
        preds[i: i + batch_size] = batch_predictions(model, batch,
                                                    seq_maxlen=seq_maxlen,
                                                    padval=padval)
    
    return preds


# ==============================================================================
#                                                                       ACCURACY
# ==============================================================================
def accuracy(a, b):
    """ Given predictions and true labels (doesnt matter which order), it
        return the accuracy.
         
    NOTE: the inputs should be only be 1 axis.
    """
    return (np.array(a) == np.array(b)).mean()


# ==============================================================================
#                                                                 EVALUATE_MODEL
# ==============================================================================
def evaluate_model(model, x, y, batch_size=128, seq_maxlen=100, padval=0):
    preds = predictions(model, x, batch_size=batch_size, seq_maxlen=seq_maxlen, padval=padval)
    return accuracy(preds, y)


# ==============================================================================
#                                                                 GET_EVALS_DICT
# ==============================================================================
def get_evals_dict(file):
    """ Loads previously saved evals dict if it exists, otherwise
        initializes a new blank one.
    """
    # KEEP TRACK OF EVALS - loading from file if they already exist
    if os.path.exists(file):
        print("LOADING EXISTING EVALS")
        evals = pickle2obj(file)
    else:
        print("INITIALIZING NEW EVALS")
        evals = {"loss": [],
                 "train_acc": [],
                 "valid_acc": [],
                 "train_time": [], # Time taken to train each round
                 "eval_time": [],  # Time on evaluation
                 "alpha": [],
                 "step": [],
                 }
    return evals


# ==============================================================================
#                                                                   UPDATE_EVALS
# ==============================================================================
def update_evals(evals, loss, train_acc, valid_acc, train_time, eval_time, alpha, step):
    evals["loss"].append(loss)
    evals["train_acc"].append(train_acc)
    evals["valid_acc"].append(valid_acc)
    evals["train_time"].append(train_time)  # Time taken to train each round
    evals["eval_time"].append(eval_time)  # Time on evaluation
    evals["alpha"].append(alpha)
    evals["step"].append(step)

