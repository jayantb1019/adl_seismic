import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import pytorch_lightning as pl 

import pdb 
import gc

import bm3d

from skimage.restoration import denoise_nl_means ,denoise_wavelet , estimate_sigma


import torchmetrics
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

import sys 
sys.path.append('../utils')

from adl_loss import Loss_L1

def bm3d_batch(noisy) : 
    
    denoised = torch.zeros_like(noisy)
    for B in range(noisy.shape[0]) : 
        patch = noisy[B,0,:,:].numpy()
        sigma = estimate_sigma(patch)
        denoised[B,0,:,:] = torch.from_numpy(bm3d.bm3d(patch, sigma_psd=sigma, stage_arg = bm3d.BM3DStages.ALL_STAGES))
    return denoised 


def nlm_batch(noisy) : 

    denoised = torch.zeros_like(noisy)
    for B in range(noisy.shape[0]) : 
        patch = noisy[B,0,:,:].numpy()
        sigma = estimate_sigma(patch)
        denoised[B,0,:,:] = torch.from_numpy(denoise_nl_means(patch, sigma=sigma, channel_axis=None))
    return denoised 


def wavelet_batch(noisy) : 
    denoised = torch.zeros_like(noisy)
    for B in range(noisy.shape[0]) : 
        patch = noisy[B,0,:,:].numpy()
        sigma = estimate_sigma(patch)
        denoised[B,0,:,:] = torch.from_numpy(denoise_wavelet(patch, sigma=sigma, channel_axis=None))
    return denoised 


class Conventional_Methods(pl.LightningModule) : 
    def __init__(self,config) : 
        
        super().__init__()
        data = config['train']['data']
        
        self.patch_size = data['patch_size']


        # reflection padding for tests
        self.reflection_pad = nn.ReflectionPad2d(self.patch_size // 2)
        self.hflipper = torchvision.transforms.RandomHorizontalFlip(p=1)   

        self.bm3d = torchvision.transforms.Lambda(bm3d_batch)
        self.nlm = torchvision.transforms.Lambda(nlm_batch)
        self.wav = torchvision.transforms.Lambda(wavelet_batch)

        self.dummy_model = torch.nn.Sequential(torch.nn.Conv2d(1,1,1,1))

        self.save_hyperparameters()


    def forward(self,x) : 

        dummy_output = self.dummy_model(x)

        return self.bm3d(x), self.nlm(x) , self.wav(x)

              


    def training_step(self, batch, batch_idx, *args, **kwargs):
        
        clean, noisy, _ = batch 

        return 1 
    
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        
        clean, noisy, _ = batch

        return 1
    
    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr = 0.001)


    def test_step(self, batch, batch_idx, *args, **kwargs) : 

        clean, noisy, _ = batch 
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)

        denoised_bm3d , denoised_nlm, denoised_wav = self(noisy)


        noisy_psnr = peak_signal_noise_ratio(noisy, clean) # let's not give data range
        noisy_ssim = structural_similarity_index_measure(noisy, clean, sigma=0.5, kernel_size = 5, )
        noisy_mae = Loss_L1(clean, noisy)

        self.log('noisy_psnr', noisy_psnr)
        self.log('noisy_ssim', noisy_ssim)
        self.log('noisy_mae', noisy_mae)

        self.calc_metrics_logs('bm3d', clean, denoised_bm3d)
        self.calc_metrics_logs('nlm', clean, denoised_nlm)
        self.calc_metrics_logs('wav', clean, denoised_wav)


    def calc_metrics_logs(self,type,clean, denoised) : 

        test_psnr = peak_signal_noise_ratio(denoised, clean)
        test_ssim = structural_similarity_index_measure(denoised, clean, sigma=0.5, kernel_size = 5, )
        test_mae = Loss_L1(clean, denoised)


        self.log(f'test_psnr_{type}', test_psnr)
        self.log(f'test_ssim_{type}', test_ssim)
        self.log(f'test_mae_{type}', test_mae) 
