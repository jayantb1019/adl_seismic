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
from pytorch_lightning.callbacks import RichModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import sys 
sys.path.append('../models')
sys.path.append('../datamodules')

from dncnn import DnCNNLightning
from dm_faciesmark import FaciesMarkDataModule

accelerator = 'cuda'
fast_dev_run = False

def get_config(config_path) : 
    # read config file 
    with open(config_path, 'r') as f : 
        config = yaml.safe_load(f)
    f.close()
    return config

def main(args) : 
    pl.seed_everything(42)
    traceback.install()
    
    CONFIG_PATH = '/content/adl_seismic/config/config_dncnn.yaml'
    
    config = get_config(CONFIG_PATH)
    
    if args['accelerator'] : 
        accelerator = args['accelerator']
        
    if args['e'] : # epochs , if provided through cmdline
        config['train']['epochs'] = args['e']
        
        
    limit_train_batches = args['lt']
    limit_val_batches = args['lv']

    modelSummaryCb = RichModelSummary(max_depth=-1)
    tqdmProgressCb = TQDMProgressBar(refresh_rate=20)
    modelCheckpointCb = ModelCheckpoint(save_top_k = 5, monitor = 'val_ssim', mode='max')
    
    
    # logger 
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    experiment_version = f"dncnn_{timestamp}"
    dncnn_logger = TensorBoardLogger('../lightning_logs', name='dncnn', log_graph=True, version = experiment_version)
    
    datamodule = FaciesMarkDataModule(config['train']['data'])

