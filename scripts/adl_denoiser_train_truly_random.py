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


import pytorch_lightning as pl 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import RichModelSummary, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger

import sys 
sys.path.append('../models')
sys.path.append('../datamodules')

from adl import Efficient_U, Efficient_U_DISC, ADL
from dm_faciesmark import FaciesMarkDataModule

# CONFIG_PATH = '../config/config_adl_faciesmark.yaml'
# CONFIG_PATH = '/Users/jayanthboddu/Desktop/data_science/upgrad/MSDS/experiments_feb/config/config_adl_faciesmark.yaml'

accelerator = 'cuda'
fast_dev_run = False

def get_config(config_path) : 
    # read config file 
    with open(config_path, 'r') as f : 
        config = yaml.safe_load(f)
    f.close()
    return config


def main(args=None) : 

    pl.seed_everything(42)
    traceback.install()
    

    CONFIG_PATH = '/local1/workspace/adl_seismic/config/config_adl_faciesmark.yaml'
    
    config = get_config(CONFIG_PATH)
    
    ##############################################
    config['train']['data']['noise_mode'] = 'truly_random'
    #################################################

    modelSummaryCb = RichModelSummary(max_depth=-1)
    tqdmProgressCb = TQDMProgressBar(refresh_rate=20)
    modelCheckpointCb = ModelCheckpoint(save_top_k = 5, monitor = 'val_ssim', mode='max')
    
    # logger 
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    experiment_version = f"adl_truly_random_{timestamp}"
    denoiser_logger = TensorBoardLogger('../lightning_logs', name='denoiser', log_graph=True, version = experiment_version)
    
    datamodule = FaciesMarkDataModule(config['train']['data'])
    
    print('''
        ================
        DENOISER WARM UP
        ================
        ''')
    
    denoiser = Efficient_U(config)
    
    cbs = [modelSummaryCb, tqdmProgressCb , modelCheckpointCb , StochasticWeightAveraging(swa_lrs=1e-2)]

        
    denoiser_trainer = pl.Trainer(
        accelerator = accelerator,
        devices=1, 
        callbacks = cbs,
        logger = denoiser_logger,
        max_epochs=config['train']['denoiser']['epochs'], 
        fast_dev_run=fast_dev_run,          
        enable_model_summary=False,
        log_every_n_steps = 5,
        # precision=32
        track_grad_norm=2, # this will plot norm-2 to tensorboard, if its increasing, then the gradients would explode.
        detect_anomaly = True, # detects nans in forward / backward pass and stops training
        gradient_clip_val=0.5, 
        resume_from_checkpoint = '/local1/workspace/adl_seismic/lightning_logs/denoiser/adl_truly_random_09_03_2023_20_01_56/checkpoints/epoch=29-step=37740.ckpt'
    )
    
    
    denoiser_trainer.fit(denoiser, datamodule)
    denoiser_trainer.test(denoiser, datamodule)
    

if __name__ == '__main__' : 
    args = dict()
    main(args)