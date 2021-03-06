################################################################################
#                                                             IMPORT AND GLOBALS
################################################################################
from __future__ import print_function, division, unicode_literals
import glob
import os
import numpy as np
import argparse

from vocab import get_vocab
from data import get_data, limit_data_vocab

from file_support import file2str, obj2pickle, pickle2obj
from support import str2ids, idtensor2str, ids2str
from support import Timer, pretty_time
from support import create_random_batch, batch_from_indices
from support import load_snapshot, take_snapshot, epoch_snapshot, load_latest_snapshot
from eval import evaluate_model, get_evals_dict, update_evals
from support import load_hyper_params, save_hyper_params

DATA_DIR = "aclImdb"
from model import Model

CACHED_DATA = "data.pickle"
# COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument("name", type=str, help="Model Name")
parser.add_argument("-n", type=int, required=True, help="Num train steps")
opt = parser.parse_args()
MODEL_NAME = opt.name
n_epochs = opt.n


# FILE AND DIR PATHS
ROOT_DIR = ""
VOCAB_FILE = os.path.join(ROOT_DIR, "vocab.txt")
MODELS_DIR = os.path.join(ROOT_DIR, "models", MODEL_NAME)
SNAPSHOTS_DIR = os.path.join(MODELS_DIR, "snapshots")
EVALS_FILE = os.path.join(MODELS_DIR, "evals.pickle")
HYPERPARAMS_FILE = os.path.join(MODELS_DIR, "hyperparams.txt")

################################################################################
#                                                           SUPPORTING FUNCTIONS
################################################################################


# ==============================================================================
#                                                                     TRAIN_STEP
# ==============================================================================
def train_step(model, X, Y):
    """ Given a model, the input X representing a batch of  sequences  of
        words, and the output labels Y, it performs a full training step,
        updating the model parameters. based on the loss and optimizer of
        the model.

    Args:
        model:      (Model object)
                    The model containing the neural net architecture.
        X:          (torch Variable)
                    The input tensor of shape: [sequence_length]
        Y:          (torch Variable)
                    The output labels tensor of shape: [1]

    Returns:
        Returns the average loss over the batch of sequences.
    """
    # Ensure model is in training mode
    if not model.training:
        model.train(True)
    
    # Get dimensions of input and target labels
    batch_size, sample_length = X.size()
    
    # Initialize hidden state and reset the accumulated gradients
    hidden = model.init_hidden(batch_size)
    model.zero_grad()
    
    # Run through the sequence on the LSTM
    output, _ = model(X, hidden)
    
    # Calculate gradients, and update parameter weights
    loss = model.loss_func(output, Y)
    loss.backward()
    model.optimizer.step()
    
    # Return the loss
    return loss.data[0]


# ==============================================================================
#                                                                  TRAIN_N_STEPS
# ==============================================================================
def train_n_steps(model, data, hyper, evals, n_steps, batch_size=128, print_every=100, eval_every=1000):
    # TODO: Start epoch timer at last epoch time from evals (but take into
    # acount eval time)
    # TODO: Do not use globals, feed paths dictionary or object.
    epoch_loss = 0
    snapshot_count = len(evals["loss"])
    start_step = evals["step"][-1] if snapshot_count > 0 else 0
    start_step += 1

    epoch_timer = Timer()
    step_timer = Timer()
    eval_timer = Timer()
    
    for step in range(1, n_steps + 1):
        step_timer.start()
        X, Y = create_random_batch(data["xtrain"], data["ytrain"],
                                   batchsize=batch_size, maxlen=hyper["SAMPLE_LENGTH"])
        loss = train_step(model, X, Y)
        epoch_loss += loss
        
        # PRINTOUTS
        if step % print_every == 0:
            progress = 100 * float(step) / n_steps
            print_train_feedback(start_step+step, loss=loss, progress=progress,
                                 elapsed_time=epoch_timer.elapsed(),
                                 avg_time_ms=step_timer.elapsed()/batch_size)
        
        # EVALUATIONS AND SNAPSHOTS
        if (step % eval_every == 0):
            epoch_time = epoch_timer.elapsed()
            
            print("=" * 60)
            snapshot_count += 1
            epoch_loss /= eval_every
            
            # EVALUATE - on train and validation data
            eval_timer.start()
            train_acc = evaluate_model(model, data["xtrain"], data["ytrain"],
                                       seq_maxlen=100)
            eval_time = eval_timer.elapsed()
            valid_acc = evaluate_model(model, data["xvalid"], data["yvalid"],
                                       seq_maxlen=100)
            print_epoch_feedback(train_acc, valid_acc, epoch_loss)
            
            # UPDATE EVALS
            update_evals(evals,
                         loss=epoch_loss,
                         train_acc=train_acc,
                         valid_acc=valid_acc,
                         train_time=epoch_time,
                         eval_time=eval_time,
                         alpha=model.alpha,
                         step=start_step+step)

            # SAVE SNAPSHOTS - of model parameters, and evals dict
            epoch_snapshot(model, snapshot_count, accuracy=valid_acc,
                           name=MODEL_NAME, dir=SNAPSHOTS_DIR)
            obj2pickle(evals, EVALS_FILE)
            save_hyper_params(hyper, HYPERPARAMS_FILE)

            # RESET
            epoch_loss = 0
            epoch_timer.start()
            print("=" * 60)
    print("DONE")


# ==============================================================================
#                                                           PRINT_TRAIN_FEEDBACK
# ==============================================================================
def print_train_feedback(step, loss, progress, elapsed_time, avg_time_ms):
    """ Prints a line of feedback about the training process such as:

         3000 (  18.8%) 00:00:02 | AVG_MS:  25.47 | LOSS:  2.42164

          ^      ^       ^         ^                ^
          |      |       |         |                L Loss
          |      |       |         |
          |      |       |         L Average train time per sample
          |      |       |
          |      |       L Elapsed time
          |      |
          |      L Progress
          |
          L Step number
    """
    # avg_time_ms = avg_time * 1000
    
    #    3000 (  18.8%) 00:00:02 | AVG_MS:  25.47 | LOSS:  2.421
    template = "{: 8d} ({: 6.1f}%) {} | AVG_MS:{: 7.2f} | LOSS:{: 3.5f}"
    print(template.format(step, progress, pretty_time(elapsed_time), avg_time_ms, loss))


# ==============================================================================
#                                                           PRINT_TRAIN_FEEDBACK
# ==============================================================================
def print_epoch_feedback(train_acc, valid_acc, loss):
    """ Prints feedback like:
        
        TRAIN ACC:  90.906 VALID ACC:  91.181  LOSS:  0.02215
    """
    s = "TRAIN ACC: {: 3.3f} VALID ACC: {: 3.3f}  LOSS: {: 3.5f}"
    print(s.format(100 * train_acc, 100 * valid_acc, loss))


################################################################################
#                                                                           DATA
################################################################################
hyper = load_hyper_params(HYPERPARAMS_FILE)

# LOAD VOCAB
# TODO: make vocab files contain FULL vocab from imdb
#       And make get_vocab() load only the first MAX_VOCAB words
id2word, word2id = get_vocab(VOCAB_FILE, DATA_DIR, hyper["MAX_VOCAB"])
n_words = len(id2word)

# CLASS MAPPINGS
id2class = ["neg", "pos"]
class2id = {label:id for id, label in enumerate(id2class)}

# LOAD DATA
data = get_data(DATA_DIR, CACHED_DATA, vocab_file=VOCAB_FILE)
limit_data_vocab(data, n=hyper["MAX_VOCAB"], unknown_id=1)
n_samples = len(data["xtrain"])


################################################################################
#                                                                          MODEL
################################################################################
model = Model(n_vocab=n_words,
              embed_size=hyper["EMBED_SIZE"],
              h_size=hyper["N_HIDDEN"],
              n_layers=hyper["N_LAYERS"],
              dropout=hyper["DROPOUT"],
              l2=hyper["L2"])
model.update_learning_rate(hyper["LAST_ALPHA"])
load_latest_snapshot(model, SNAPSHOTS_DIR)



################################################################################
#                                                                          TRAIN
################################################################################
# Load evaluations dictionary
evals = get_evals_dict(EVALS_FILE)

# Calculate important steps
steps_per_epoch = int(np.ceil(n_samples / hyper["BATCH_SIZE"]))
n_steps = n_epochs * steps_per_epoch

# TRAIN - and handle early termination through keyboard interrupt
try:
    print("#" * 60)
    print(MODEL_NAME.upper())
    print("#" * 60)
    train_n_steps(model, data, evals,
                  n_steps=n_steps,
                  batch_size=hyper["BATCH_SIZE"],
                  print_every=int(steps_per_epoch / 5),
                  eval_every=steps_per_epoch)
except KeyboardInterrupt:
    print("\nEARLY TERMINATION INITIATED - saving evals and hypers")
    obj2pickle(evals, EVALS_FILE)
    save_hyper_params(hyper, HYPERPARAMS_FILE)
    
# TODO: Pad the data with zeroes at the end not begining.
# TODO: Create plots and save them in each epoch
# TODO: Custome weight initializations  - See  https://discuss.pytorch.org/t/weight-initilzation/157/9
# TODO: Time in training printout is based on epoch not total time since train_n_steps called


