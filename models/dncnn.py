import numpy as np 
import pdb 

import torch 
import torch.nn as nn 
import torch.nn.functional as F


from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio
import pytorch_lightning as pl 


import sys 
sys.path.append('/local1/workspace/seismic_image_enhancement/githubrepos/KAIR-master/models')

from network_dncnn import DnCNN

class DnCNNLightning(pl.LightningModule) : 
    def __init__(self, training_config) :
        super().__init__()
        training_config = training_config['train']
        self.batch_size = training_config['data']['batch_size']
        self.patch_size = training_config['data']['patch_size']
        self.lr = training_config['dncnn']['lr']
        self.b1 = training_config['dncnn']['b1']
        self.b2 = training_config['dncnn']['b2']

        self.nlayers = training_config['dncnn']['nlayers']
        self.noise_mode = training_config['data']['noise_mode']
        self.noise_factor = training_config['data']['noise_factor']
        
        self.model = DnCNN(in_nc = 1, out_nc=1, nc=64, nb=self.nlayers, act_mode ='IL') # instance normalisation and leaky relu
        
        # self.loss_function = torch.nn.MSELoss() if (training_config['loss_function'] == 'l2') else torch.nn.L1Loss()
        
        self.save_hyperparameters() 
        
        self.example_input_array = torch.zeros(self.batch_size, 1, self.patch_size, self.patch_size).double()
        
        
    def forward(self, x) : 
        return self.model(x) # ( model output is noise )
    
    def training_step(self, batch, batch_idx) : 
        clean, noisy, _ = batch 
        noise = self(noisy)
        
        loss = self.loss_function(noisy - noise, clean)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx) : 
        clean, noisy, _ = batch 
        noise = self(noisy)
        
        loss = self.loss_function(noisy - noise, clean)
        self.log('val_loss', loss, prog_bar=True)
        
        with torch.no_grad() : 
            psnr = peak_signal_noise_ratio(noisy - noise, clean)
            ssim = structural_similarity_index_measure(noisy-noise, clean)
            
            self.log('val_psnr', psnr)
            self.log('val_ssim', ssim)
        
        # return loss
    
    def test_step(self,batch, batch_idx) : 
        clean, noisy, _ = batch 
        noise = self(noisy)
        
        loss = self.loss_function(noisy - noise, clean)
        self.log('test_loss', loss)
        
        with torch.no_grad() : 
            psnr = peak_signal_noise_ratio(noisy - noise, clean)
            ssim = structural_similarity_index_measure(noisy-noise, clean)
            
            self.log('test_psnr', psnr)
            self.log('test_ssim', ssim)
        
        # return loss
    
    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        optimiser = torch.optim.Adam(self.parameters(), lr = lr, betas=(b1, b2))
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience = 3, threshold=0.001, verbose=True)

        return {
            'optimizer' : optimiser, 
            'lr_scheduler' : {
                'scheduler' : lr_scheduler, 
                'monitor'   : 'val_loss',
                'interval'  : 'epoch',
                'frequency' : 1
            }
        }

    def on_train_epoch_end(self) -> None:
            
        # visualising w and b 
        for name, params in self.named_parameters() : 
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
        