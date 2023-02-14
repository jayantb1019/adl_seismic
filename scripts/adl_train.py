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
# CONFIG_PATH = '/Users/jayanthboddu/Desktop/data_science/upgrad/MSDS/experiments_feb/config/config_adl_faciesmark.yaml'

device = 'cuda'
fast_dev_run = False

def get_config(config_path) : 
    # read config file 
    with open(config_path, 'r') as f : 
        config = yaml.safe_load(f)
    f.close()
    return config


def main(bs) : 
    pl.seed_everything(42)
    traceback.install()
    
    config = get_config(CONFIG_PATH)
    
    if bs : # override config with argparse inputs
        config['train']['data']['batch_size'] = bs
    
    modelSummaryCb = RichModelSummary(max_depth=-1)
    tqdmProgressCb = TQDMProgressBar(refresh_rate=20)
    
    # logger 
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    experiment_version = f"adl_{timestamp}"
    denoiser_logger = TensorBoardLogger('../lightning_logs', name='denoiser', log_graph=False, version = experiment_version)
    discriminator_logger = TensorBoardLogger('../lightning_logs', name='discriminator', log_graph=False, version = experiment_version)
    adl_logger = TensorBoardLogger('../lightning_logs', name='adl', log_graph=False, version = experiment_version)
    
    datamodule = FaciesMarkDataModule(config['train']['data'])
    
    
    
    
    
    # PHASE 1 : 
    # print('''
    #       ================
    #       DENOISER WARM UP
    #       ================
    #       ''')
    # denoiser = Efficient_U(config)
    
    # denoiser_trainer = pl.Trainer(
    #     accelerator = device,
    #     devices=1, 
    #     callbacks = [modelSummaryCb, tqdmProgressCb ],
    #     logger = denoiser_logger,
    #     max_epochs=config['train']['denoiser']['epochs'], 
    #     fast_dev_run=fast_dev_run,          
    #     enable_model_summary=False,
    #     # precision=32
    # )
    
    # # pdb.set_trace()
    
    # denoiser_trainer.fit(denoiser, datamodule)
    
    # PHASE 2 : 
    # print('''
    #       =====================
    #       DISCRIMINATOR WARM UP
    #       =====================
    #       ''')

    # discriminator_trainer = pl.Trainer(
    #     accelerator = device,
    #     devices=1, 
    #     callbacks = [modelSummaryCb, tqdmProgressCb ],
    #     logger = discriminator_logger,
    #     max_epochs=config['train']['discriminator']['epochs'], 
    #     fast_dev_run=fast_dev_run, 
    #      enable_model_summary=False,      
    #      # precision=32   
    # )
    
    # denoiser_checkpoint_path = '/content/denoiser_20230213_epoch=49-step=27600.ckpt'
    
    # trained_denoiser = Efficient_U(config).load_from_checkpoint(denoiser_checkpoint_path)
    # discriminator = Efficient_U_DISC(trained_denoiser, config)
    
    
    # discriminator_trainer.fit(discriminator, datamodule)
    
    # pdb.set_trace()
    
    # PHASE 3 : 
    print('''
          =====================
              ADL TRAINING
          =====================
          ''')
    
    adl_trainer = pl.Trainer(
        accelerator = device,
        devices=1, 
        callbacks = [modelSummaryCb, tqdmProgressCb ],
        logger = adl_logger,
        max_epochs=config['train']['ADL']['epochs'], 
        fast_dev_run=fast_dev_run, 
        enable_model_summary=False,        
        # precision=32 
    )
    denoiser_checkpoint_path = '/content/denoiser_20230213_epoch=49-step=27600.ckpt'
    discriminator_checkpoint_path = '/content/discriminator_20230213_epoch=49-step=27600.ckpt'
    
    pdb.set_trace()
    
    trained_denoiser = Efficient_U(config).load_from_checkpoint(denoiser_checkpoint_path)
    trained_discriminator = Efficient_U_DISC(trained_denoiser, config).load_from_checkpoint(discriminator_checkpoint_path)

    adl = ADL(trained_denoiser, trained_discriminator, config)
    
    adl_trainer.fit(adl, datamodule) 
    
    pdb.set_trace()
    

if __name__ == '__main__' : 
    
    parser = ArgumentParser()
    parser.add_argument('-bs', type=int, default=128)
    
    bs = parser.parse_args().bs
    main(bs)