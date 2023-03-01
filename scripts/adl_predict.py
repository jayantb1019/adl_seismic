from argparse import ArgumentParser
import os 
from glob import glob
import time 
from datetime import datetime 

import pdb 
import yaml 
import numpy as np 
from tqdm import tqdm

import torch 
import torch.nn as nn 
from torchvision.transforms import RandomHorizontalFlip

from rich import traceback 
import gc 

torch.cuda.empty_cache()

import pytorch_lightning as pl 
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import RichModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A

import sys 
sys.path.append('../models')
sys.path.append('../datamodules')

from adl import Efficient_U, Efficient_U_DISC, ADL
from dm_faciesmark import FaciesMarkDataModule

sys.path.append('../utils/')
from data import patch_index_generation, select_patch
from augmentations import polarity_reversal, random_trace_dropout, random_high_noise_trace, random_amp_attenuation, random_trace_shuffle, horizontal_flip, random_amp_shift, rotate_patch

from skimage.util import random_noise
from skimage.filters import butterworth

# config getter
def get_config(config_path) : 
    # read config file 
    with open(config_path, 'r') as f : 
        config = yaml.safe_load(f)
    f.close()
    return config

def load_model(config) : 
    # config
    accelerator = config['predict']['accelerator']

    # paths
    denoiser_ckpt_path = '/local1/workspace/adl_seismic/lightning_logs/denoiser/adl_27_02_2023_18_16_16/checkpoints/epoch=59-step=27360.ckpt'
    discriminator_ckpt_path = '/local1/workspace/adl_seismic/lightning_logs/discriminator/adl_28_02_2023_12_34_11/checkpoints/epoch=10-step=5016.ckpt'
    adl_ckpt_path = '/local1/workspace/adl_seismic/lightning_logs/adl/adl_28_02_2023_14_23_58/checkpoints/epoch=82-step=830.ckpt'

    with torch.no_grad() : 
        trained_denoiser = Efficient_U(config).load_from_checkpoint(denoiser_ckpt_path).eval()
        # trained_discriminator = Efficient_U_DISC.load_from_checkpoint(checkpoint_path = discriminator_ckpt_path, model=trained_denoiser, config=config).eval()


        # trained_adl_model = ADL.load_from_checkpoint(checkpoint_path = adl_ckpt_path, denoiser=trained_denoiser, discrminator=trained_discriminator, config=config).eval()

        trained_denoiser.to(accelerator)
        # trained_discriminator.to(accelerator)
        # trained_adl_model.to(accelerator)
    return trained_denoiser



def load_data(config, test_data_path) : # assuming it is a numpy array in ILINE,TWT,XLINE format

    test_data = np.load(test_data_path)

    accelerator = config['test']['accelerator']

    test_data = test_data.transpose([0,2,1])

    return torch.from_numpy(test_data).to(accelerator)



def predict(model, data, config) : 
    accelerator = config['test']['accelerator']

    data = data.to(accelerator)

    denoised, _, _ = model(data)
    denoised = denoised.clamp(-1,1)
    return denoised


def pad_(data, factor = 8) : 
    '''Check if the dimensions of data are a multiple of 32. If not, pad , also add reflection padding = half of data size to remove any edge effects'''

    b,c,h,w = data.shape 

    h_, w_ = h ,w

    if h % factor : 
        h_ = ( h// factor + 1 ) * factor

    if w % factor : 
        w_ = ( w// factor + 1 ) * factor

    
    pad = torch.zeros((b,c, h_,w_))

    pad[:,:,:h, :w] = data 

    refpad = nn.ReflectionPad2d((w_//2 , w_//2 , h_//2 , h_//2 ))

    return refpad(pad), h, w 

def unpad_(data, h, w) : # original h,w 

    b,c, h_refpad, w_refpad = data.shape 

    data_unrefpad = data[:, :, h_refpad//4 : h_refpad//4 + h, w_refpad//4 : w_refpad // 4 + w ]


    return data_unrefpad
        
    
def denoise(model, noisy_iline, config) :

    accelerator = config['test']['accelerator'] 
    noisy_iline = noisy_iline.to(accelerator)

    noisy_iline_padded , h, w = pad_(noisy_iline)
    denoised_padded = predict(model, noisy_iline_padded, config)
    denoised = unpad_(denoised_padded,h,w)

    return denoised

    


def main(args=None) : 

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
    test_data_path = '../data/faciesmark/test_once/test2_seismic.npy'

    # Test time augmentations 

    # model 
    model = load_model(config)

    # data
    noisy = load_data(config, test_data_path) # tensor with dims (iline, twt, xline)
    denoised =torch.zeros_like(noisy)

    for i in tqdm(range(noisy.shape[0]), desc='Denoising ilines') :

        noisy_c = torch.unsqueeze(torch.unsqueeze(noisy[i], 0),0)

        # TEST TIME AUGMENTATIONS
        noisy_c_pr = -1 * noisy_c # polarity reversal 

        # horizontal_flipper 
        hflipper = RandomHorizontalFlip(p=1).to('cuda')

        noisy_c_flipped = hflipper(noisy_c) # horizontal flipping

        noisy_pr_flipped = -1 * noisy_c_flipped # horizontal flipping and polarity reversal 

        

        denoised_c = denoise(model, noisy_c, config)   


        denoised_c_pr = -1 * denoise(model, noisy_c_pr, config)
        denoised_c_flipped = hflipper(denoise(model, noisy_c_flipped, config)) # flip back after denoising 
        denoised_c_pr_flipped = -1 * hflipper(denoise(model, noisy_pr_flipped, config)) # flip back and reverse polarise


        denoised[i] = (torch.squeeze(denoised_c + denoised_c_pr + denoised_c_flipped + denoised_c_pr_flipped) /4.0)

        noisy_ = None 
        noisy_c = None 
        noisy_c_pr = None 
        noisy_c_flipped = None 
        noisy_pr_flipped = None 
        noisy_pr_flipped = None 
        denoised_c = None 
        denoised_c_pr = None 
        denoised_c_flipped = None 
        denoised_c_pr_flipped = None 

        torch.cuda.empty_cache()
        gc.collect()

    pdb.set_trace()

    # predict 

if __name__ ==  '__main__' :
    parser = ArgumentParser()
    parser.add_argument('-loc', type=str, default='workstation') # or workstation 

    parsed_args = parser.parse_args()

    args = dict(loc=parsed_args.loc)

    main(args)



    


