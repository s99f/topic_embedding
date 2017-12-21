__author__ = 'xiaozhi'
import os
import sys
import random
from multiprocessing import cpu_count
import logging as log
import numpy as np

ori = ""

for line in sys.stdin:
    ll = line.strip().split()
