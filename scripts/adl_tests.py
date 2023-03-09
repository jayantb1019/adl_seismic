import torch 
import torch.nn as nn 
import yaml 
from argparse import ArgumentParser

from rich import traceback

import pytorch_lightning as pl 

import sys 
sys.path.append('../models')
sys.path.append('../datamodules')

from adl import Efficient_U, Efficient_U_DISC, ADL
from dm_faciesmark import FaciesMarkDataModule

def get_config(config_path) : 
    # read config file 
    with open(config_path, 'r') as f : 
        config = yaml.safe_load(f)
    f.close()
    return config

def main() : 
    pl.seed_everything(42)
    parser = ArgumentParser()
    parser.add_argument('-nt', type=str, default='gaussian') # noise type, could be gaussian, poisson, mixed, lpf
    parser.add_argument('-nl', type=float, default = 0.01)
    
    pargs = parser.parse_args()
    
    nt = pargs.nt 
    nl = pargs.nl
    
    print(f'''
          Noise Level : {nl}
          Noise Type  : {nt}
          ''')
    
    
    
    
    CONFIG_PATH = '../config/final_config_interpretation.yaml'
    
    
    config = get_config(CONFIG_PATH)
    
    if nt : 
        config['train']['data']['noise_mode'] = nt
    
    if nl : 
        config['train']['data']['noise_factor'] = nl
        

    denoiser_trainer = pl.Trainer(
        accelerator = 'mps')
    
    
    denoiser_checkpoint_path = '../lightning_logs/final/denoiser/adl_final_07_03_2023_16_28_38_gaussian_0.01/checkpoints/epoch=49-step=31100.ckpt'
    trained_denoiser = Efficient_U.load_from_checkpoint(checkpoint_path = denoiser_checkpoint_path, config = config)
    datamodule = FaciesMarkDataModule(config['train']['data'])
    
    denoiser_trainer.test(trained_denoiser, datamodule)
    
    
if __name__ == '__main__' : 
    main()