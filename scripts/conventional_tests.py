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

from conventional import Conventional_Methods
from dm_faciesmark import FaciesMarkDataModule

# model config
CONFIG_MODEL = '../config/final_config_conventional.yaml'

# dataset folders 
CONFIG_PATH_INTERPRETATION = '../config/final_config_interpretation.yaml'
CONFIG_PATH_FACIESMARK = '../config/final_config_faciesmark.yaml'
CONFIG_PATH_STDATA12 = '../config/final_config_stdata12.yaml'

# results file path 
RESULTS_FILE_PATH = '../results/conventional_results_20230311_v2.csv'

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
    parser.add_argument('-dataset', type=str, default = 'faciesmark')
    pargs = parser.parse_args()
    
    tnt = pargs.tnt 
    tnl = pargs.tnl
    dataset = pargs.dataset

    print(f'''
          ===============================
          Test Dataset : {dataset}

          Test Noise Level : {tnl}
          Test Noise Type  : {tnt}

          ===============================
          ''')
    
     # TEST DATASET SELECTION
    
    if dataset == 'faciesmark' : 
        CONFIG_PATH = CONFIG_PATH_FACIESMARK
    
    elif dataset == 'interpretation' : 
        CONFIG_PATH =CONFIG_PATH_INTERPRETATION

    elif dataset == 'stdata12' : 
        CONFIG_PATH = CONFIG_PATH_STDATA12   

    ds_config = get_config(CONFIG_PATH)
    model_config = get_config(CONFIG_MODEL)

    # pass noise args 
    ds_config['train']['data']['noise_mode'] = tnt
    ds_config['train']['data']['noise_factor'] = tnl 

    # logger 
    conv_logger = TensorBoardLogger('../lightning_logs', name='covnentional', log_graph=True, version = timestamp)

    denoiser_trainer = pl.Trainer(accelerator = 'cpu', logger=conv_logger)

    conventional_class = Conventional_Methods(model_config)

    datamodule = FaciesMarkDataModule(ds_config['train']['data'])

    gc.collect()
    results = denoiser_trainer.test(conventional_class, datamodule)[0]
    gc.collect()
    # write to text file 

    with open(RESULTS_FILE_PATH, 'a') as f : 
        f.writelines(f"{timestamp},{dataset},{tnt},{tnl},{results['noisy_mae']},{results['noisy_psnr']},{results['noisy_ssim']},{results['test_mae_bm3d']},{results['test_psnr_bm3d']},{results['test_ssim_bm3d']},{results['test_mae_nlm']},{results['test_psnr_nlm']},{results['test_ssim_nlm']},{results['test_mae_wav']},{results['test_psnr_wav']},{results['test_ssim_wav']}\n")
    f.close()


if __name__ == '__main__' : 
    main()
