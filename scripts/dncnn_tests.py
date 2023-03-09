#!/root/anaconda3/envs/lightning/bin/python

import torch 
import torch.nn as nn 
import yaml 
from argparse import ArgumentParser
from datetime import datetime 
import gc
import pdb 

from rich import traceback

import pytorch_lightning as pl 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichModelSummary

import sys 
sys.path.append('../models')
sys.path.append('../datamodules')

from dncnn import DnCNNLightning
from dm_faciesmark import FaciesMarkDataModule

# model config
CONFIG_MODEL = '../config/config_dncnn.yaml'

# dataset folders 
CONFIG_PATH_INTERPRETATION = '../config/final_config_interpretation.yaml'
CONFIG_PATH_FACIESMARK = '../config/final_config_faciesmark.yaml'
CONFIG_PATH_STDATA12 = '../config/final_config_stdata12.yaml'

# checkpoint paths 
CKPT_PATH_G_01 = '/local1/workspace/adl_seismic/checkpoints/dncnn/dncnn_08_03_2023_gaussian_0.01/checkpoints/epoch=19-step=16600.ckpt'
CKPT_PATH_G_05 = '/local1/workspace/adl_seismic/checkpoints/dncnn/dncnn_08_03_2023_gaussian_0.05/checkpoints/epoch=17-step=14940.ckpt'
CKPT_PATH_G_1 = '/local1/workspace/adl_seismic/checkpoints/dncnn/dncnn_02_03_2023_gaussian_0.1/checkpoints/epoch=49-step=24800.ckpt'
CKPT_PATH_G_5 = '/local1/workspace/adl_seismic/checkpoints/dncnn/dncnn_02_03_2023_gaussian_0.5/checkpoints/epoch=49-step=24800.ckpt'
CKPT_PATH_P = '/local1/workspace/adl_seismic/checkpoints/dncnn/dncnn_08_03_2023_poisson/checkpoints/epoch=17-step=14940.ckpt'
CKPT_PATH_MIXED = '/local1/workspace/adl_seismic/checkpoints/dncnn/dncnn_08_03_2023_mixed/checkpoints/epoch=17-step=14940.ckpt'
CKPT_PATH_LPF = '/local1/workspace/adl_seismic/checkpoints/dncnn/dncnn_08_03_2023_lpf/checkpoints/epoch=17-step=14940.ckpt'

# results file path 
RESULTS_FILE_PATH = '/local1/workspace/adl_seismic/results/dncnn_results.csv'

def get_config(config_path) : 
    # read config file 
    with open(config_path, 'r') as f : 
        config = yaml.safe_load(f)
    f.close()
    return config

def main() : 
    pl.seed_everything(42)

    torch.cuda.empty_cache()

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

    
    ds_config = get_config(CONFIG_PATH)
    model_config = get_config(CONFIG_MODEL)

    if tnt : 
        ds_config['train']['data']['noise_mode'] = tnt
    
    if tnl : 
        ds_config['train']['data']['noise_factor'] = tnl

    if (CKPT_PATH == CKPT_PATH_G_1) or (CKPT_PATH == CKPT_PATH_G_5) : 
        model_config['train']['dncnn']['bias'] = True
    
        

    denoiser_logger = TensorBoardLogger('../lightning_logs', name='dncnn_tests', log_graph=True, version = timestamp)
    modelSummaryCb = RichModelSummary(max_depth=-1)

    denoiser_trainer = pl.Trainer(
        accelerator = 'cuda', callbacks = [modelSummaryCb], logger=denoiser_logger)
    
    
    trained_dncnn = DnCNNLightning.load_from_checkpoint(checkpoint_path = CKPT_PATH, training_config = model_config, strict=False)
    trained_dncnn = trained_dncnn.eval()

    datamodule = FaciesMarkDataModule(ds_config['train']['data'])
    
    gc.collect()
    results = denoiser_trainer.test(trained_dncnn, datamodule)[0]
    gc.collect()

    # pdb.set_trace()

    # write to text file 

    with open(RESULTS_FILE_PATH, 'a') as f : 
        f.writelines(f"{timestamp},{dataset},{cnt},{cnl},{tnt},{tnl},{results['noisy_mae']},{results['noisy_psnr']},{results['noisy_ssim']},{results['test_mae']},{results['test_psnr']},{results['test_ssim']}\n")
    f.close()

    
    
if __name__ == '__main__' : 
    main()