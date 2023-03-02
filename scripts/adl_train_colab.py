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

def get_checkpoint(model_type, experiment_version, config) : 
    
    base_folder = config['train']['data']['dir']['logs_folder']
    folder_path = os.path.join(base_folder, model_type, experiment_version, 'checkpoints')

    ckpt = glob(folder_path + '/*.ckpt')

    return ckpt[0]


def main(args) : 
    pl.seed_everything(42)
    traceback.install()
    
    if args['loc'] == 'colab' : 
        CONFIG_PATH = '/content/adl_seismic/config/config_adl_faciesmark_colab.yaml'
    elif args['loc'] == 'workstation' : 
        CONFIG_PATH = '/local1/workspace/adl_seismic/config/config_adl_faciesmark.yaml'
    
    elif args['loc'] == 'kaggle' : 
        CONFIG_PATH = '/kaggle/working/adl_seismic/config/config_adl_faciesmark_kaggle.yaml'

    elif args['loc'] == 'win' : 
        CONFIG_PATH = r'Z:\experiments_feb\config\config_adl_faciesmark_win.yaml'
    
    config = get_config(CONFIG_PATH)
    
    if args['bs'] : # override config with argparse inputs
        config['train']['data']['batch_size'] = args['bs']
        
    if args['accelerator'] : 
        accelerator = args['accelerator']

    if args['e'] : # epochs , if provided through cmdline
        config['train']['denoiser']['epochs'] = args['e']
        config['train']['discriminator']['epochs'] = args['e']
        config['train']['ADL']['epochs'] = args['e']
        
        
    if args['lambda1'] : 
        config['train']['ADL']['lambda1'] = args['lambda1']


    limit_train_batches = args['lt']
    limit_val_batches = args['lv']

    modelSummaryCb = RichModelSummary(max_depth=-1)
    tqdmProgressCb = TQDMProgressBar(refresh_rate=20)
    modelCheckpointCb = ModelCheckpoint(save_top_k = 5, monitor = 'val_ssim', mode='max')
    
    # logger 
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    experiment_version = f"adl_{timestamp}"
    denoiser_logger = TensorBoardLogger('../lightning_logs', name='denoiser', log_graph=True, version = experiment_version)
    discriminator_logger = TensorBoardLogger('../lightning_logs', name='discriminator', log_graph=True, version = experiment_version)
    adl_logger = TensorBoardLogger('../lightning_logs', name='adl', log_graph=True, version = experiment_version)
    
    datamodule = FaciesMarkDataModule(config['train']['data'])
    
    # pdb.set_trace()

    # trial : 21.02.2023 just a denoiser test 
    # denoiser performance test # uncomment to run denoiser baseline
    # denoiser_ckpt_path_test = '/local1/workspace/adl_seismic/lightning_logs/denoiser/adl_20_02_2023_12_50_02/checkpoints/epoch=49-step=2750.ckpt'
    # trained_denoiser = Efficient_U(config).load_from_checkpoint(denoiser_ckpt_path_test)
    # denoiser_trainer = pl.Trainer(
    #     accelerator = accelerator,
    #     devices=1, 
    #     callbacks = [modelSummaryCb, tqdmProgressCb ],
    #     logger = denoiser_logger,
    #     max_epochs=config['train']['denoiser']['epochs'], 
    #     fast_dev_run=fast_dev_run,          
    #     enable_model_summary=False,
    #     limit_train_batches = limit_train_batches, 
    #     limit_val_batches = limit_val_batches
    #     # precision=32, 
    #     track_grad_norm=2, # this will plot norm-2 to tensorboard, if its increasing, then the gradients would explode.
    #     detect_anomaly = True # detects nans in forward / backward pass and stops training
    # )

    # denoiser_trainer.test(trained_denoiser, datamodule)
    # pdb.set_trace()

    #PHASE 1 : 
    print('''
          ================
          DENOISER WARM UP
          ================
          ''')
    
    denoiser = Efficient_U(config)
    
    if args['loc'] == 'kaggle' : 
        denoiser_checkpoint_path = '/kaggle/working/adl_seismic/checkpoints/denoiser.ckpt' # colab
    if args['loc'] == 'workstation' : 
        denoiser_checkpoint_path = '/local1/workspace/adl_seismic/adl_seismic/lightning_logs/denoiser/adl_21_02_2023_15_09_14_best/checkpoints/epoch=49-step=27300.ckpt' # colab
    if args['loc'] == 'colab' : 
        denoiser_checkpoint_path = '/content/drive/MyDrive/adl_seismic/lightning_logs/denoiser/adl_21_02_2023_15_09_14_best/checkpoints/epoch=49-step=27300.ckpt'
        
    # denoiser_checkpoint_path = '/local1/workspace/adl_seismic/lightning_logs/denoiser/adl_27_02_2023_18_16_16/checkpoints/epoch=59-step=27360.ckpt' # workstation
    # print(f"Denoiser Checkpoint : {denoiser_checkpoint_path}")
    # trained_denoiser = Efficient_U(config).load_from_checkpoint(denoiser_checkpoint_path)
    denoiser_trainer = pl.Trainer(
        accelerator = accelerator,
        devices=1, 
        callbacks = [modelSummaryCb, tqdmProgressCb , modelCheckpointCb, StochasticWeightAveraging(swa_lrs=1e-2) ],
        logger = denoiser_logger,
        max_epochs=config['train']['denoiser']['epochs'], 
        fast_dev_run=fast_dev_run,          
        enable_model_summary=False,
        limit_train_batches = limit_train_batches, 
        limit_val_batches = limit_val_batches, 
        log_every_n_steps = 5,
        # precision=32
        track_grad_norm=2, # this will plot norm-2 to tensorboard, if its increasing, then the gradients would explode.
        detect_anomaly = True, # detects nans in forward / backward pass and stops training
        gradient_clip_val=0.5
    )
    
    
    denoiser_trainer.fit(denoiser, datamodule)
    # denoiser_trainer.test(denoiser, datamodule)
    # denoiser_trainer.test(trained_denoiser, datamodule)

    # pdb.set_trace()

    # print(results)
    
    
    
    # PHASE 2 : 
    # print('''
    #       =====================
    #       DISCRIMINATOR WARM UP
    #       =====================
    #       ''')

    discriminator_trainer = pl.Trainer(
        accelerator = accelerator,
        devices=1, 
        callbacks = [modelSummaryCb, tqdmProgressCb ],
        logger = discriminator_logger,
        max_epochs=config['train']['discriminator']['epochs'], 
        fast_dev_run=fast_dev_run, 
         enable_model_summary=False,      
        limit_train_batches = limit_train_batches, 
        limit_val_batches = limit_val_batches,
        log_every_n_steps= 5, 
        # overfit_batches= 10,
        # check_val_every_n_epoch = 5
         # precision=32   
        track_grad_norm=2, # this will plot norm-2 to tensorboard, if its increasing, then the gradients would explode.
        detect_anomaly = True, # detects nans in forward / backward pass and stops training
        gradient_clip_val=0.5
    )
    
    # denoiser_checkpoint_path = '/local1/workspace/adl_seismic/lightning_logs/denoiser/adl_20_02_2023_12_50_02/checkpoints/epoch=49-step=2750.ckpt'
    
    # denoiser_checkpoint_path = '/local1/workspace/adl_seismic/lightning_logs/denoiser/adl_27_02_2023_14_32_03/checkpoints/epoch=49-step=250.ckpt'
    
    # denoiser_checkpoint_path = get_checkpoint('denoiser',experiment_version, config)
    trained_denoiser = Efficient_U(config).load_from_checkpoint(denoiser_checkpoint_path)
    
    # pdb.set_trace()
    
    
    discriminator = Efficient_U_DISC(trained_denoiser, config)
    
    
    # discriminator_trainer.fit(discriminator, datamodule) 
    # denoiser_trainer.test(trained_denoiser, datamodule)
    
    # denoiser_trainer.test(trained_denoiser, datamodule)
    
    # pdb.set_trace()
    
    # PHASE 3 : 
    print('''
          =====================
              ADL TRAINING
          =====================
          ''')
    
    adl_trainer = pl.Trainer(
        accelerator = accelerator,
        devices=1, 
        callbacks = [modelSummaryCb, tqdmProgressCb, modelCheckpointCb  ],
        logger = adl_logger,
        max_epochs=config['train']['ADL']['epochs'], 
        fast_dev_run=fast_dev_run, 
        enable_model_summary=False,       
        limit_train_batches = limit_train_batches, 
        limit_val_batches = limit_val_batches,
        log_every_n_steps = 5,   
        # overfit_batches=10,
        track_grad_norm=2, # this will plot norm-2 to tensorboard, if its increasing, then the gradients would explode.
        detect_anomaly = True, # detects nans in forward / backward pass and stops training
        gradient_clip_val=0.5
        # check_val_every_n_epoch = 5
        # precision=32 
    )
    
    
    
    # denoiser_checkpoint_path = get_checkpoint('denoiser', experiment_version, config)
    discriminator_checkpoint_path = get_checkpoint('discriminator', experiment_version, config)
    
    # pdb.set_trace()
    
    # denoiser_checkpoint_path = '/Users/jayanthboddu/Desktop/data_science/upgrad/MSDS/experiments_feb/lightning_logs/denoiser_20230213_epoch=49-step=27600.ckpt'
    #discriminator_checkpoint_path = '/local1/workspace/adl_seismic/lightning_logs/discriminator/adl_28_02_2023_15_25_21/epoch=10-step=5016_saved.ckpt'
    # print(f"Discrminator Checkpoint :{discriminator_checkpoint_path}")
    # pdb.set_trace()
    
    # trained_denoiser = Efficient_U.load_from_checkpoint(checkpoint_path = denoiser_checkpoint_path, config = config)
    trained_discriminator = Efficient_U_DISC.load_from_checkpoint(checkpoint_path = discriminator_checkpoint_path, model=trained_denoiser, config=config) 
    # trained_discriminator = Efficient_U_DISC.(model=trained_denoiser, config=config)

    adl = ADL(trained_denoiser, trained_discriminator, config)
    
    # adl_trainer.fit(adl, datamodule) 
    # adl_trainer.test(adl, datamodule) 
    
    # pdb.set_trace()
    

if __name__ == '__main__' : 
    
    parser = ArgumentParser()
    parser.add_argument('-bs', type=int, default=128)
    parser.add_argument('-acc', type=str, default='cuda')
    parser.add_argument('-loc', type=str, default='colab') # or workstation 
    parser.add_argument('-lt', type=float, default = 0.99) # limit train batches
    parser.add_argument('-lv', type=float, default = 0.99) # limit val batches
    parser.add_argument('-e', type=int, default=None)
    parser.add_argument('-lambda1', type=int, default = None )

    
    bs = parser.parse_args().bs
    accelerator = parser.parse_args().acc
    loc = parser.parse_args().loc 
    lt = parser.parse_args().lt
    lv = parser.parse_args().lv
    e = parser.parse_args().e
    lambda1 = parser.parse_args().lambda1
    
    args = dict(bs = bs , accelerator = accelerator, loc=loc , lt = lt, lv = lv, e = e, lambda1= lambda1)
    main(args)