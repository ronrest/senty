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

