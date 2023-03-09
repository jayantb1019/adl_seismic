#!/root/anaconda3/envs/lightning/bin/python

import torch 
import torch.nn as nn 
import yaml 
from argparse import ArgumentParser
from datetime import datetime 
import gc
import pdb 

import numpy as np

torch.cuda.empty_cache()

from rich import traceback

import pytorch_lightning as pl 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichModelSummary

import sys 
sys.path.append('../models')
sys.path.append('../datamodules')

from adl import Efficient_U, Efficient_U_DISC, ADL
from dm_faciesmark import FaciesMarkDataModule

# dataset folders 
CONFIG_PATH_INTERPRETATION = '../config/final_config_interpretation.yaml'
CONFIG_PATH_FACIESMARK = '../config/final_config_faciesmark.yaml'
CONFIG_PATH_STDATA12 = '../config/final_config_stdata12.yaml'

# checkpoint paths 
CKPT_PATH_G_01 = '/local1/workspace/adl_seismic/checkpoints/denoiser/adl_final_07_03_2023_16_28_38_gaussian_0.01/checkpoints/epoch=49-step=31100.ckpt'
CKPT_PATH_G_05 = '/local1/workspace/adl_seismic/checkpoints/denoiser/adl_final_07_03_2023_20_16_53_gaussian_0.05/checkpoints/epoch=49-step=15550.ckpt'
CKPT_PATH_G_1 = '/local1/workspace/adl_seismic/checkpoints/denoiser/adl_final_08_03_2023_03_06_17_gaussian_0.1/checkpoints/epoch=49-step=15550.ckpt'
CKPT_PATH_G_5 = '/local1/workspace/adl_seismic/checkpoints/denoiser/adl_final_08_03_2023_03_13_19_gaussian_0.5/checkpoints/epoch=48-step=15239.ckpt'
CKPT_PATH_P = '/local1/workspace/adl_seismic/checkpoints/denoiser/adl_final_08_03_2023_10_37_42_poisson/checkpoints/epoch=49-step=15550.ckpt'
CKPT_PATH_MIXED = '/local1/workspace/adl_seismic/checkpoints/denoiser/adl_final_08_03_2023_07_59_41_mixed/checkpoints/epoch=49-step=15550.ckpt'
CKPT_PATH_LPF = '/local1/workspace/adl_seismic/checkpoints/denoiser/adl_final_08_03_2023_06_34_49_lpf/checkpoints/epoch=49-step=15550.ckpt'

# results file path 
RESULTS_FILE_PATH = '/local1/workspace/adl_seismic/results/adl_results.csv'

def get_config(config_path) : 
    # read config file 
    with open(config_path, 'r') as f : 
        config = yaml.safe_load(f)
    f.close()
    return config

def main() : 
    pl.seed_everything(42)



    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    parser = ArgumentParser()
    parser.add_argument('-tnt', type=str, default='gaussian') # noise type, could be gaussian, poisson, mixed, lpf
    parser.add_argument('-tnl', type=float, default = 0.01)
    parser.add_argument('-cnt', type=str, default='gaussian') # noise type, could be gaussian, poisson, mixed, lpf
    parser.add_argument('-cnl', type=float, default = 0.01)
    parser.add_argument('-dataset', type=str, default = 'faciesmark')
    pargs = parser.parse_args()
    
    tnt = pargs.tnt 
    tnl = pargs.tnl

    cnt = pargs.cnt 
    cnl = pargs.cnl 

    dataset = pargs.dataset
    
    print(f'''
          ===============================
          Test Dataset : {dataset}

          Test Noise Level : {tnl}
          Test Noise Type  : {tnt}

          Checkpoint Noise Level : {cnl}
          Checkpoint Noise Type : {cnt}

          ===============================
          ''')
    

    
    # TEST DATASET SELECTION
    
    if dataset == 'faciesmark' : 
        CONFIG_PATH = CONFIG_PATH_FACIESMARK
    
    elif dataset == 'interpretation' : 
        CONFIG_PATH =CONFIG_PATH_INTERPRETATION

    elif dataset == 'stdata12' : 
        CONFIG_PATH = CONFIG_PATH_STDATA12
    
    # CHECKPOINT SELECTION 

    CKPT_PATH = None 
    if cnt == 'gaussian' : 
        if cnl == 0.01 : 
            CKPT_PATH = CKPT_PATH_G_01
        if cnl == 0.1 : 
            CKPT_PATH = CKPT_PATH_G_1
        if cnl == 0.5 : 
            CKPT_PATH = CKPT_PATH_G_5
        if cnl == 0.05 : 
            CKPT_PATH = CKPT_PATH_G_05

    if cnt == 'mixed' : 
        CKPT_PATH = CKPT_PATH_MIXED
    if cnt == 'poisson' : 
        CKPT_PATH = CKPT_PATH_P
    if cnt == 'lpf' : 
        CKPT_PATH = CKPT_PATH_LPF

    
    config = get_config(CONFIG_PATH)
    
    if tnt : 
        config['train']['data']['noise_mode'] = tnt
    
    if tnl : 
        config['train']['data']['noise_factor'] = tnl
        

    denoiser_logger = TensorBoardLogger('../lightning_logs', name='denoiser', log_graph=True, version = timestamp)
    modelSummaryCb = RichModelSummary(max_depth=-1)

    denoiser_trainer = pl.Trainer(
        accelerator = 'cuda', callbacks = [modelSummaryCb], logger=denoiser_logger)
    
    
    trained_denoiser = Efficient_U.load_from_checkpoint(checkpoint_path = CKPT_PATH, config = config)
    trained_denoiser = trained_denoiser.eval()
    
    gc.collect()
    noisy_iline = '/local1/workspace/random_denoising/data/noisy_inline_samples/noisy_iline_mp41b_pstm_rnd_gaussian_0.1.npy'
    iline_np = np.load(noisy_iline)

    iline_tensor = torch.from_numpy(iline_np)

    results = denoiser_trainer.predict(trained_denoiser, iline_tensor)[0]

    pdb.set_trace()
    gc.collect()

    
if __name__ == '__main__' : 
    main()