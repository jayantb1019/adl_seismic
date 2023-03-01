from argparse import ArgumentParser
import os 
from glob import glob
import time 
from datetime import datetime 

import pdb

import yaml


import torch 
import torch.nn as nn 

from rich import traceback

torch.cuda.empty_cache()
# torch.set_float32_matmul_precision('medium') # doesnt work on MX600

import pytorch_lightning as pl 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

import sys 
sys.path.append('../models')
sys.path.append('../datamodules')

from dncnn import Efficient_U, Efficient_U_DISC, ADL
from dm_faciesmark import FaciesMarkDataModule