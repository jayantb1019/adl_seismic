from argparse import ArgumentParser
import os 
import time 
from datetime import datetime 

import pdb

import yaml


import torch 
import torch.nn as nn 

from rich import traceback

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

import sys 
sys.path.append('../models')
sys.path.append('../datamodules')

from adl import Efficient_U, Efficient_U_DISC, ADL
from dm_faciesmark import FaciesMarkDataModule

CONFIG_PATH = '/content/adl_seismic/config/config_adl_faciesmark.yaml'

def get_config(config_path) : 
    # read config file 
    with open(config_path, 'r') as f : 
        config = yaml.safe_load(f)
    f.close()
    return config


def main() : 
    pl.seed_everything(42)
    traceback.install()
    
    config = get_config(CONFIG_PATH)
    
    modelSummaryCb = RichModelSummary(max_depth=-1)
    tqdmProgressCb = TQDMProgressBar(refresh_rate=20)
    
    # logger 
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    experiment_version = f"adl_{timestamp}"
    denoiser_logger = TensorBoardLogger('lightning_logs', name='denoiser', log_graph=False, version = experiment_version)
    discriminator_logger = TensorBoardLogger('lightning_logs', name='discriminator', log_graph=False, version = experiment_version)
    adl_logger = TensorBoardLogger('lightning_logs', name='adl', log_graph=False, version = experiment_version)
    
    datamodule = FaciesMarkDataModule(config['train']['data'])
    
    
    
    
    
    # PHASE 1 : 
    denoiser = Efficient_U(config)
    
    denoiser_trainer = pl.Trainer(
        accelerator = 'cuda',
        devices=1, 
        callbacks = [modelSummaryCb, tqdmProgressCb ],
        logger = denoiser_logger,
        max_epochs=config['train']['denoiser']['epochs'], 
        fast_dev_run=True,          
        enable_model_summary=False,
        # precision=32
    )
    
    
    denoiser_trainer.fit(denoiser, datamodule)
    
    # PHASE 2 : 

    discriminator_trainer = pl.Trainer(
        accelerator = 'cuda',
        devices=1, 
        callbacks = [modelSummaryCb, tqdmProgressCb ],
        logger = discriminator_logger,
        max_epochs=config['train']['discriminator']['epochs'], 
        fast_dev_run=True, 
         enable_model_summary=False,      
         # precision=32   
    )
    trained_denoiser = Efficient_U(config).load_from_checkpoint('best')
    discriminator = Efficient_U_DISC(config, trained_denoiser)
    
    discriminator_trainer.fit(discriminator, datamodule)
    
    # PHASE 3 : 
    
    adl_trainer = pl.Trainer(
        accelerator = 'cuda',
        devices=1, 
        callbacks = [modelSummaryCb, tqdmProgressCb ],
        logger = adl_logger,
        max_epochs=config['train']['ADL']['epochs'], 
        fast_dev_run=True, 
        enable_model_summary=False,        
        # precision=32 
    )
    
    trained_denoiser = Efficient_U(config).load_from_checkpoint('best')
    trained_discriminator = Efficient_U_DISC(config, trained_denoiser).load_from_checkpoint('best')

    adl = ADL(trained_denoiser, trained_discriminator, config)
    
    adl_trainer.fit(adl, datamodule) 
    
    pdb.set_trace()
    

if __name__ == '__main__' : 
    main()