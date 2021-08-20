from multiprocessing.spawn import freeze_support

from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.tabular import *
import numpy as np
import torch
from fastai import train
import cflearn

import time


if __name__ == '__main__':

    cflearn.make().fit()



