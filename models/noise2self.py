'''
Ref : Batson and Royer paper 

'''

import pdb

import sys 
sys.path.append('../utils')
# sys.path.append('../github_repos/noise2self-master/')
# sys.path.append('../github_repos/noise2self-master/models/')
sys.path.append('../github_repos/KAIR-master/models')

import numpy as np

import torch
import torch.nn as nn
from torch.nn import L1Loss 
from torch.optim import Adam


import pytorch_lightning as pl
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio

from segy_input import  get_config , load_seisnc

from skimage.util import random_noise

from skimage import data, img_as_float

from mask import Masker

from dncnn import DnCNN


class Noise2Self(pl.LightningModule) : 

    def __init__(self, config) : 

        super().__init__()

        self.config = config 
        self.patch_size = config['patch_based_training']['patch_size']
        self.dncnn = DnCNN(in_nc=1, out_nc=1,nc=64, nb = config['dncnn']['num_layers'], act_mode='IL').double()
        self.masker = Masker(width = config['dncnn']['mask_width'] , mode='interpolate')

        self.loss_function = L1Loss()
        self.denoised = None

        self.best_validation_loss = 1

        self.save_hyperparameters()

        self.example_input_array = torch.zeros(1,1, self.patch_size, self.patch_size ).double()


    def forward(self, x) : 

        
        nn_output = self.dncnn(x)

        return nn_output

    def training_step(self, batch, batch_idx) : 
        noisy = batch 

        mask_index = self.global_step % (self.masker.n_masks - 1)

        nn_input , mask = self.masker.mask(noisy, mask_index)

        nn_output = self(nn_input)
        
        loss = self.loss_function(nn_output * mask , noisy * mask)

        self.log('train_loss', loss, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx) : 

        noisy = batch 

        total_mask_loss = 0

        for i in range(self.masker.n_masks) : 

            net_input, mask = self.masker.mask(noisy, self.masker.n_masks - 1)

            net_output = self(net_input)

            loss = self.loss_function(net_output*mask , noisy*mask)

            total_mask_loss += loss

        loss = total_mask_loss / self.masker.n_masks # average loss on all masks

        self.log('val_loss', loss,prog_bar=True)

        with torch.no_grad() : 

            psnr = peak_signal_noise_ratio(torch.clip(net_output, -1, 1), noisy)

            self.log('val_psnr', psnr,prog_bar=True)

            ssim = structural_similarity_index_measure(torch.clip(net_output, -1, 1), noisy)

            self.log('val_ssim', ssim,prog_bar=True)

            if loss < self.best_validation_loss : 
                self.best_validation_loss = loss 
                self.denoised = torch.clip(self(noisy).detach(), -1, 1)


        # return loss
    


    def configure_optimizers(self) : 

        lr = self.config['dncnn']['learning_rate'][0]
        optimiser = torch.optim.Adam(self.parameters(), lr = lr,betas=[0.9,0.999], eps=1e-7)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', patience=5, threshold=0.001, verbose=True)
        return {
            'optimizer' : optimiser, 
            'lr_scheduler' : {
                'scheduler' : lr_scheduler, 
                'monitor'   : 'val_psnr',
                'interval'  : 'epoch',
                'frequency' : 1
            }
        }

    def on_train_epoch_end(self) : 
        for name, params in self.named_parameters(): 
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    # def preprocess(self, iline) : 

    #     abs_max = abs(iline).quantile(0.99) # take the 99th percentile abs value

    #     iline = iline.clip(-abs_max, abs_max) # clip

    #     iline_norm = iline / abs_max

    #     return iline_norm 

    #     # TODO : add a mask to get trace wise extents where data is missing 
    #     # byte locations 
    #     # 

#     def add_noise(data, mode, mean,noise_factor) : 

#         args = dict(mode = mode, seed=42, clip=True, mean =mean , var = noise_factor)
#         return random_noise(data, **args)



#     def create_noisy_iline(self, seisnc, iline_no, noise_factor, noise_mode) : 

#         iline = seisnc.sel(iline=iline_no).transpose('twt','xline')

#         iline_norm = iline.map(preprocess)

#         mean_ = 0 

#         iline_norm_noisy = iline_norm.map(add_noise, args=(noise_mode,mean_, noise_factor) )
        
#         return iline_norm, iline_norm_noisy


# def create_clean_noisy_patch(iline_norm, iline_norm_noisy, xline_start_index=1078-512-200, twt_start_index=500, patch_size=512) : 

#     # # taking one 512x512 patch
#     # patch_size = 512
#     # twt_start_index = 500
#     # xline_start_index = 1078 - 512 - 200

#     clean_patch = iline_norm.isel(twt=slice(twt_start_index , twt_start_index + patch_size), xline = slice(xline_start_index, xline_start_index + patch_size))
#     noisy_patch = iline_norm_noisy.isel(twt=slice(twt_start_index , twt_start_index + patch_size), xline = slice(xline_start_index, xline_start_index + patch_size))
    
#     clean_patch = img_as_float(clean_patch.compute().data)
#     noisy_patch = img_as_float(noisy_patch.compute().data) 

#     noisy_patch = torch.Tensor(noisy_patch[np.newaxis, np.newaxis])

#     return noisy_patch, clean_patch 

    

if __name__  == '__main__' : 

    CONFIG_PATH = '/local1/workspace/random_denoising/config/config_noise2self.yaml'
    config  = get_config(CONFIG_PATH)
    model = Noise2Self(config)


    CLEAN_SEISNC_PATH = '/local1/workspace/random_denoising/data/mp41b_pstm_stk_rnd/MP41B_PSTM_STK_RND.seisnc'
    seisnc = load_seisnc(CLEAN_SEISNC_PATH)
    iline_no = 1155
    device = 'cuda'
    noise_factor = config['patch_based_training']['noise_factors'][-1]
    noise_mode = config['patch_based_training']['noise_modes'][0]

    twt_start_index = 500 
    patch_size= config['patch_based_training']['patch_size']
    xline_start_index = 1078-512-200

    # iline_norm, iline_norm_noisy = create_noisy_iline(seisnc, iline_no, noise_factor, noise_mode)
    # noisy_patch, clean_patch = create_clean_noisy_patch(iline_norm, iline_norm_noisy, xline_start_index, twt_start_index, patch_size)


    # model = model.to(device)
    # noisy = noisy_patch.to(device)

    # denoised = model(noisy)

