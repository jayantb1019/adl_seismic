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


# torch.set_float32_matmul_precision('medium') # doesnt work on MX600

import pytorch_lightning as pl 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import RichModelSummary, ModelCheckpoint,StochasticWeightAveraging
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
    
    # CONFIG_PATH = '/content/adl_seismic/config/config_dncnn.yaml'
    CONFIG_PATH = '../config/config_dncnn.yaml'
    
    config = get_config(CONFIG_PATH)
    
    if args['accelerator'] : 
        accelerator = args['accelerator']
        if accelerator == 'cuda' : 
            torch.cuda.empty_cache()
        
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
    
    
    dncnn_trainer = pl.Trainer(
        accelerator = accelerator,
        devices=1, 
        callbacks = [modelSummaryCb, tqdmProgressCb , modelCheckpointCb, StochasticWeightAveraging(swa_lrs=1e-2) ],
        logger = dncnn_logger,
        max_epochs=config['train']['epochs'], 
        fast_dev_run=fast_dev_run,          
        enable_model_summary=False,
        limit_train_batches = limit_train_batches, 
        limit_val_batches = limit_val_batches, 
        log_every_n_steps = 5,
        precision=16,
        track_grad_norm=2, # this will plot norm-2 to tensorboard, if its increasing, then the gradients would explode.
        detect_anomaly = True, # detects nans in forward / backward pass and stops training
        gradient_clip_val=0.5
    )

    dncnn_model = DnCNNLightning(config)
    dncnn_trainer.fit(dncnn_model, datamodule)
    dncnn_trainer.test(dncnn_model, datamodule)
    
if __name__ == '__main__' : 
    
    parser = ArgumentParser()
    parser.add_argument('-bs', type=int, default=128)
    parser.add_argument('-acc', type=str, default='cuda')
    parser.add_argument('-loc', type=str, default='colab') # or workstation 
    parser.add_argument('-lt', type=float, default = 0.99) # limit train batches
    parser.add_argument('-lv', type=float, default = 0.99) # limit val batches
    parser.add_argument('-e', type=int, default=None)

    
    bs = parser.parse_args().bs
    accelerator = parser.parse_args().acc
    loc = parser.parse_args().loc 
    lt = parser.parse_args().lt
    lv = parser.parse_args().lv
    e = parser.parse_args().e

    args = dict(bs = bs , accelerator = accelerator, loc=loc , lt = lt, lv = lv, e = e)
    main(args)