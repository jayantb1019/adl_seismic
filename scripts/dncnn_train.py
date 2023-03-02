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
from pytorch_lightning.callbacks import RichModelSummary, ModelCheckpoint,EarlyStopping
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

    if args['nt'] : 
        config['train']['data']['noise_type'] = args['nt']
    
    if args['nl'] : 
        config['train']['data']['noise_factor'] = args['nl']

    if args['nlayers'] : 
        config['train']['dncnn']['nlayers'] = args['nlayers']

    if args['act'] :
        config['train']['dncnn']['act'] = args['act']

    if args['w'] :
        config['train']['dncnn']['w'] = args['w']  

    if args['bias'] != None :
        config['train']['dncnn']['bias'] = args['bias'] 

    if args['loc'] == 'colab' : 
        config['train']['data']['dir']['data_root'] = '/content/adl_seismic/data/faciesmark'

    elif args['loc'] == 'workstation' : 
        config['train']['data']['dir']['data_root'] = '/local1/workspace/adl_seismic/data/faciesmark'

    

    limit_train_batches = args['lt']
    limit_val_batches = args['lv']

    modelSummaryCb = RichModelSummary(max_depth=-1)
    tqdmProgressCb = TQDMProgressBar(refresh_rate=20)
    modelCheckpointCb = ModelCheckpoint(save_top_k = 5, monitor = 'val_ssim', mode='max')
    earlyStoppingCb = EarlyStopping(monitor='val_ssim',mode='max', patience=10)
    
    
    # logger 
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    noise_mode = config['train']['data']['noise_type'] 
    noise_level = config['train']['data']['noise_factor'] 
    experiment_version = f"dncnn_epochs_{config['train']['epochs']}_lt_{limit_train_batches}_{noise_mode}_{noise_level}_{timestamp}"
    dncnn_logger = TensorBoardLogger('../lightning_logs', name='dncnn', log_graph=True, version = experiment_version)
    
    datamodule = FaciesMarkDataModule(config['train']['data'])
    
    
    dncnn_trainer = pl.Trainer(
        accelerator = accelerator,
        devices=1, 
        callbacks = [modelSummaryCb, tqdmProgressCb , modelCheckpointCb, earlyStoppingCb ],
        logger = dncnn_logger,
        max_epochs=config['train']['epochs'], 
        fast_dev_run=fast_dev_run,          
        enable_model_summary=False,
        limit_train_batches = limit_train_batches, 
        limit_val_batches = limit_val_batches, 
        log_every_n_steps = 5,
        # precision=16,
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
    parser.add_argument('-nt', type=str, default='gaussian') # noise type, could be gaussian, poisson, mixed, lpf
    parser.add_argument('-nl', type=float, default = 0.5) # noise level, between 0-1. Prefer 0.1,0.5,0.9 # only works for gaussian
    parser.add_argument('-nlayers', type=int, default = 15) # no of layers 
    parser.add_argument('-w', type=int, default=64) # width / no of filters 
    parser.add_argument('-act', type=str, default='IL') # can be BL = batch norm + leaky relu or IL = Instance Norm + leaky relu
    parser.add_argument('-bias', type=bool, default=False)

    
    bs = parser.parse_args().bs
    accelerator = parser.parse_args().acc
    loc = parser.parse_args().loc 
    lt = parser.parse_args().lt
    lv = parser.parse_args().lv
    e = parser.parse_args().e
    nt = parser.parse_args().nt
    nl = parser.parse_args().nl
    nlayers = parser.parse_args().nlayers
    act = parser.parse_args().act
    w = parser.parse_args().w
    bias = parser.parse_args().bias

    args = dict(bs = bs , 
                accelerator = accelerator, 
                loc=loc , 
                lt = lt, 
                lv = lv,
                e = e, 
                nt=nt, 
                nl=nl, 
                nlayers = nlayers, 
                w = w, 
                act = act, 
                bias = bias)
    main(args)