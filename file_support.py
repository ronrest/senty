from __future__ import print_function, unicode_literals
import pickle
import glob
import os
from io import open

# ==============================================================================
#                                                                    MAYBE_MKDIR
# ==============================================================================
def maybe_mkdir(path):
    """ Checks if a directory path exists on the system, if it does not, then
        it creates that directory (and any parent directories needed to
        create that directory)
    """
    if not os.path.exists(path):
        os.makedirs(path)


# ==============================================================================
#                                                           GET_PARENT_DIRECTORY
# ==============================================================================
def get_parent_directory(file):
    """ Given a file path, it returns the parent directory of that file. """
    return os.path.dirname(file)


# ==============================================================================
#                                                              MAYBE_MAKE_PARDIR
# ==============================================================================
def maybe_make_pardir(file):
    """ Takes a path to a file, and creates the necessary directory structure
        on the system to ensure that the parent directory exists (if it does
        not already exist)
    """
    maybe_mkdir(get_parent_directory(file))


# ==============================================================================
#                                                                     OBJ2PICKLE
# ==============================================================================
def obj2pickle(obj, file, protocol=-1):
    """ Saves an object as a binary pickle file to the desired file path.

    Args:
        obj:        The python object you want to save.
        file:       (string)
                    File path of file you want to save as.  eg /tmp/myFile.pkl
        protocol:   (int)(default=-1)
                    Protocol to pass to pickle.dump()
    """
    s = file if len(file) < 41 else (file[:10] + "..." + file[-28:])
    print("Saving: ", s, end="")
    
    # maybe make the parent dir
    pardir = os.path.dirname(file)
    if not (pardir == ""):
        maybe_mkdir(pardir)
    
    with open(file, mode="wb") as fileObj:
        pickle.dump(obj, fileObj, protocol=protocol)
    
    print(" -- [DONE]")


# ==============================================================================
#                                                                     PICKLE2OBJ
# ==============================================================================
def pickle2obj(file):
    """ Takes a filepath to a pickle object, and returns a python object
        specified by that pickle file.
    """
    s = file if len(file) < 41 else (file[:10] + "..." + file[-28:])
    print("Loading: ", s, end="")
    
    with open(file, mode="rb") as fileObj:
        obj = pickle.load(fileObj)
    
    print(" -- [DONE]")
    return obj


# ==============================================================================
#                                                                       FILE2STR
# ==============================================================================
def file2str(file, encoding="UTF-8"):
    """Takes a file path and returns the contents of that file as a string."""
    with open(file, "r", encoding=encoding) as textFile:
        text = textFile.read()
    return text


# ==============================================================================
#                                                                      LIST2FILE
# ==============================================================================
def list2file(a, file, encoding="UTF-8", verbose=True):
    """ Given a list, it saves the contents of each element in a new line """
    if verbose:
        print("Writing to file: ", file, end="")
        
    # maybe make the parent dir
    pardir = os.path.dirname(file)
    if not (pardir == ""):
        maybe_mkdir(pardir)

    with open(file, "w", encoding=encoding) as textFile:
        text = textFile.writelines([unicode(line) + "\n" for line in a])

    if verbose:
        print(" -- [DONE]")


# ==============================================================================
#                                                                      FILE2LIST
# ==============================================================================
def file2list(file, dtype=unicode, encoding="UTF-8"):
    """ Takes a file path and returns each line of the file as an element
        of a list. You can optionally typecast each line to the desired
        datatype (by default it sets things to unicode strings.
        
        NOTE: that it strips out any whitespaces for each line
    """
    with open(file, "r", encoding=encoding) as textFile:
        a = textFile.readlines()
        a = [dtype(item.strip()) for item in a]
    return a

