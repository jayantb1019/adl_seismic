import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 


import torchmetrics
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure, multiscale_structural_similarity_index_measure

import pdb 

import sys 
sys.path.append('../utils')

from adl_MODELS import Efficient_Unet_disc , Efficient_Unet
from adl_loss import Loss_L1 , Loss_PYR, Loss_Hist

#TODO : Change the milestones (list of epoch indices) in MultiStepLR schedulers.

RELU = nn.ReLU(inplace=False)
# TANH = nn.Tanh()

class Efficient_U(pl.LightningModule) : # denoiser
    def __init__(self, config) : 
        super().__init__()
        data = config['train']['data']
        
        self.patch_size = data['patch_size']
        self.batch_size = data['batch_size']
        self.drop_last = data['drop_last']
        
        
        denoiser_config = config['train']['denoiser']
        self.lr = denoiser_config['lr']
        self.optimizer = denoiser_config['optimizer']
        self.lr_scheduler = denoiser_config['lr_scheduler']['type']
        self.lr_scheduler_gamma = denoiser_config['lr_scheduler']['kwargs']['gamma']
        
        self.model = Efficient_Unet(in_ch=1, out_ch = 1, filter_base=32, bias=False)
        
        self.save_hyperparameters()
        
        self.example_input_array = torch.zeros(self.batch_size, 1, self.patch_size, self.patch_size)
        
    def forward(self, x) : 
        return self.model(x)
    
    def _rescale_gt_2d(self,im):
        """ downsacle image by factor 2 and 4 """
        im_x2 = F.interpolate(im, size=(im.shape[2]//2, im.shape[3]//2), mode='bilinear',
                    align_corners=False).clamp(min=-1.0, max=1.0)
        im_x4 = F.interpolate(im, size=(im.shape[2]//4, im.shape[3]//4), mode='bilinear',
                    align_corners=False).clamp(min=-1.0, max=1.0)
        return im_x2.to(torch.float32).to(self.device), im_x4.to(torch.float32).to(self.device)
        
        
    def training_step(self, batch, batch_idx, *args, **kwargs):
        
        clean, noisy, _ = batch 
        
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)
        
        denoised, denoised_2, denoised_4 = self(noisy)
        clean_2, clean_4 = self._rescale_gt_2d(clean)
        
        l1_loss_1 = Loss_L1(clean, denoised)
        l1_loss_2 = Loss_L1(clean_2, denoised_2)
        l1_loss_4 = Loss_L1(clean_4, denoised_4)
        
        pyr_loss_1 = Loss_PYR(clean, denoised, levels = 3)
        pyr_loss_2 = Loss_PYR(clean_2, denoised_2, levels = 3)
        pyr_loss_4 = Loss_PYR(clean_4, denoised_4, levels = 3)
        
        hist_loss_1  = Loss_Hist(clean, denoised)
        hist_loss_2  = Loss_Hist(clean_2, denoised_2)
        hist_loss_4  = Loss_Hist(clean_4, denoised_4)
        
        self.log('denoiser_train_l1_loss', l1_loss_1 + l1_loss_2 + l1_loss_4)
        self.log('denoiser_train_pyr_loss', pyr_loss_1 + pyr_loss_2 + pyr_loss_4)
        self.log('denoiser_train_hist_loss', hist_loss_1 + hist_loss_2 + hist_loss_4)
        
        train_loss = l1_loss_1 + l1_loss_2 + l1_loss_4 + pyr_loss_1 + pyr_loss_2 + pyr_loss_4 + hist_loss_1 + hist_loss_2 + hist_loss_4
        self.log('train_loss', train_loss, prog_bar=True)

        train_psnr = peak_signal_noise_ratio(denoised.detach(), clean.detach(), datarange=2.0)
        train_ssim = structural_similarity_index_measure(denoised.detach(), clean.detach(), sigma=0.5, kernel_size = 5, )
        train_msssim = multiscale_structural_similarity_index_measure(denoised.detach(), clean.detach(), sigma=0.5, kernel_size = 5, data_range=2.0)

        self.log('train_psnr', train_psnr)
        self.log('train_ssim', train_ssim)
        self.log('train_msssim', train_msssim)
        
        return train_loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        
        clean, noisy, _ = batch
        denoised, denoised_2, denoised_4 = self(noisy)
        clean_2, clean_4 = self._rescale_gt_2d(clean)
        
        l1_loss_1 = Loss_L1(clean, denoised)
        l1_loss_2 = Loss_L1(clean_2, denoised_2)
        l1_loss_4 = Loss_L1(clean_4, denoised_4)
        
        pyr_loss_1 = Loss_PYR(clean, denoised, levels = 3)
        pyr_loss_2 = Loss_PYR(clean_2, denoised_2, levels = 3)
        pyr_loss_4 = Loss_PYR(clean_4, denoised_4, levels = 3)
        
        hist_loss_1  = Loss_Hist(clean, denoised)
        hist_loss_2  = Loss_Hist(clean_2, denoised_2)
        hist_loss_4  = Loss_Hist(clean_4, denoised_4)
        
        
        self.log('denoiser_val_l1_loss', l1_loss_1 + l1_loss_2 + l1_loss_4)
        self.log('denoiser_val_pyr_loss', pyr_loss_1 + pyr_loss_2 + pyr_loss_4)
        self.log('denoiser_val_hist_loss', hist_loss_1 + hist_loss_2 + hist_loss_4)
        
        val_loss = l1_loss_1 + l1_loss_2 + l1_loss_4 + pyr_loss_1 + pyr_loss_2 + pyr_loss_4 + hist_loss_1 + hist_loss_2 + hist_loss_4
       
        self.log('val_loss', val_loss, prog_bar=True)

        val_psnr = peak_signal_noise_ratio(denoised.detach(), clean.detach(), datarange=2.0)
        val_ssim = structural_similarity_index_measure(denoised.detach(), clean.detach(), sigma=0.5, kernel_size = 5, )
        val_msssim = multiscale_structural_similarity_index_measure(denoised.detach(), clean.detach(), sigma=0.5, kernel_size = 5, data_range=2.0)

        self.log('val_psnr', val_psnr)
        self.log('val_ssim', val_ssim)
        self.log('val_msssim', val_msssim)

        return val_loss 
         
    
    def configure_optimizers(self):
        lr = self.lr

        optimiser = torch.optim.Adam(self.parameters(), lr = lr)
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
        
    def on_train_epoch_end(self) :
            
        # visualising w and b 
        for name, params in self.named_parameters() : 
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
             
class Efficient_U_DISC(pl.LightningModule) :  # discriminator
    def __init__(self, model, config,  *args, **kwargs) : 
        super().__init__()
        data = config['train']['data']
        self.patch_size = data['patch_size']
        self.batch_size = data['batch_size']
        self.drop_last = data['drop_last']
        
        disc_config = config['train']['discriminator']
        self.negative_slope = disc_config['negative_slope']
        
        self.lr = disc_config['lr']
        self.optimizer = disc_config['optimizer']
        self.lr_scheduler = disc_config['lr_scheduler']['type']
        self.lr_scheduler_gamma = disc_config['lr_scheduler']['kwargs']['gamma']
        
        
        self.save_hyperparameters(ignore=['model', 'disc_model'])
        
        self.model = model # trained denoiser model
        self.disc_model = Efficient_Unet_disc(in_ch=1, out_ch=1, negative_slope = self.negative_slope, filter_base = 16 , bias=False)
        
        self.example_input_array = torch.zeros(self.batch_size, 1, self.patch_size, self.patch_size)
        
    def forward(self, x) : 
        return self.disc_model(x)
    
    def _rescale_gt_2d(self,im):
        """ downsacle image by factor 2 and 4 """
        im_x2 = F.interpolate(im, size=(im.shape[2]//2, im.shape[3]//2), mode='bilinear',
                    align_corners=False).clamp(min=-1.0, max=1.0)
        im_x4 = F.interpolate(im, size=(im.shape[2]//4, im.shape[3]//4), mode='bilinear',
                    align_corners=False).clamp(min=-1.0, max=1.0)
        return im_x2.to(torch.float32).to(self.device), im_x4.to(torch.float32).to(self.device)
    
    def training_step(self, batch, batch_idx, *args, **kwargs) : ## look at this again!! discriminator should train on half a batch of clean and half a batch of denoised
        
        # loss for true sample
        clean, noisy, _ = batch 
        
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)
        
        
        gt_bridge, gt_x0, gt_x2, gt_x4 = self.disc_model(clean)
        B = clean.shape[0]
        
        true_ravel = torch.concat([torch.reshape(gt_bridge, (B,-1)),
                                torch.reshape(gt_x0, (B,-1)), 
                                torch.reshape(gt_x2, (B,-1)),
                                torch.reshape(gt_x4, (B,-1))
                                ], axis=-1)
        loss_true = torch.mean(RELU(1.0 - true_ravel)) 
    
    
        y = noisy
        y_bridge, y_pred, y_pred_x2, y_pred_x4 = self.disc_model(self.model(y)[0])
        B = y.shape[0]

        # Compute the loss for the true sample
        pred_ravel = torch.concat([torch.reshape(y_bridge, (B,-1)),
                            torch.reshape(y_pred, (B,-1)), 
                            torch.reshape(y_pred_x2, (B,-1)),
                            torch.reshape(y_pred_x4, (B,-1))
                            ], axis=-1)
        loss_pred = torch.mean(RELU(1.0 + pred_ravel)) 
        
        loss = (loss_true + loss_pred)/2
        
        self.log('disc_train_loss', loss, prog_bar=True )
        
        return loss
    
    def validation_step(self, batch, batch_idx, *args, **kwargs) : ## look at this again!! discriminator should train on half a batch of clean and half a batch of denoised
        
        # loss for true sample
        clean, noisy, _ = batch 
        
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)
        
        
        gt_bridge, gt_x0, gt_x2, gt_x4 = self.disc_model(clean)
        B = clean.shape[0]
        
        true_ravel = torch.concat([torch.reshape(gt_bridge, (B,-1)),
                                torch.reshape(gt_x0, (B,-1)), 
                                torch.reshape(gt_x2, (B,-1)),
                                torch.reshape(gt_x4, (B,-1))
                                ], axis=-1)
        loss_true = torch.mean(RELU(1.0 - true_ravel)) 
    
    
        y = noisy
        y_bridge, y_pred, y_pred_x2, y_pred_x4 = self.disc_model(self.model(y)[0])
        B = y.shape[0]

        # Compute the loss for the true sample
        pred_ravel = torch.concat([torch.reshape(y_bridge, (B,-1)),
                            torch.reshape(y_pred, (B,-1)), 
                            torch.reshape(y_pred_x2, (B,-1)),
                            torch.reshape(y_pred_x4, (B,-1))
                            ], axis=-1)
        loss_pred = torch.mean(RELU(1.0 + pred_ravel)) 
        
        loss = (loss_true + loss_pred)/2
        
        self.log('disc_val_loss', loss, prog_bar=True )
        
        return loss
    
    

    def configure_optimizers(self):
        
        lr = self.lr

        optimiser = torch.optim.Adam(self.parameters(), lr = lr)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, 
        #                                                     milestones = list(range(15,50,5)), 
        #                                                     gamma=self.lr_scheduler_gamma, verbose=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', patience = 5, threshold=0.001, verbose=True)

        return {
            'optimizer' : optimiser, 
            'lr_scheduler' : {
                'scheduler' : lr_scheduler, 
                'monitor'   : 'disc_val_loss',
                'interval'  : 'epoch',
                'frequency' : 1
            }
        }
        
    def on_train_epoch_end(self) :
            
        # visualising w and b 
        for name, params in self.named_parameters() : 
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
        
class ADL(pl.LightningModule) : # Full ADL model
    def __init__(self, denoiser, discriminator, config) : 
        super().__init__() 
        self.denoiser = denoiser 
        self.discriminator = discriminator 
        
        data_config = config['train']['data']
        self.patch_size = data_config['patch_size']
        self.batch_size = data_config['batch_size']
        
        adl_config = config['train']['ADL']
        self.lr = adl_config['lr']
        self.optimizer = adl_config['optimizer']
        self.lr_scheduler = adl_config['lr_scheduler']['type']
        self.gamma = adl_config['lr_scheduler']['kwargs']['gamma']
        
        self.save_hyperparameters(ignore=['denoiser', 'discriminator'])
        self.example_input_array = torch.zeros(self.batch_size, 1, self.patch_size, self.patch_size)
        
    def forward(self, x) : 
        denoised = self.denoiser(x)
        discriminator_output = self.discriminator(denoised[0]) # for model summary
        return denoised
    
    def _rescale_gt_2d(self,im):
        """ downsacle image by factor 2 and 4 """
        im_x2 = F.interpolate(im, size=(im.shape[2]//2, im.shape[3]//2), mode='bilinear',
                    align_corners=False).clamp(min=-1.0, max=1.0)
        im_x4 = F.interpolate(im, size=(im.shape[2]//4, im.shape[3]//4), mode='bilinear',
                    align_corners=False).clamp(min=-1.0, max=1.0)
        return im_x2.to(torch.float32).cuda(self.device), im_x4.to(torch.float32).cuda(self.device)
    
    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs) :
        
        clean, noisy, _ = batch 
        
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)
        
        
        if optimizer_idx == 0 : # denoiser 
            
            denoised, denoised_2, denoised_4 = self.denoiser(noisy)
            clean_2, clean_4 = self._rescale_gt_2d(clean)
            
            l1_loss_1 = Loss_L1(clean, denoised)
            l1_loss_2 = Loss_L1(clean_2, denoised_2)
            l1_loss_4 = Loss_L1(clean_4, denoised_4)
            
            pyr_loss_1 = Loss_PYR(clean, denoised, levels = 3)
            pyr_loss_2 = Loss_PYR(clean_2, denoised_2, levels = 3)
            pyr_loss_4 = Loss_PYR(clean_4, denoised_4, levels = 3)
            
            hist_loss_1  = Loss_Hist(clean, denoised)
            hist_loss_2  = Loss_Hist(clean_2, denoised_2)
            hist_loss_4  = Loss_Hist(clean_4, denoised_4)
            
            self.log('denoiser_train_l1_loss', l1_loss_1 + l1_loss_2 + l1_loss_4)
            self.log('denoiser_train_pyr_loss', pyr_loss_1 + pyr_loss_2 + pyr_loss_4)
            self.log('denoiser_train_hist_loss', hist_loss_1 + hist_loss_2 + hist_loss_4)
            
            train_loss = l1_loss_1 + l1_loss_2 + l1_loss_4 + pyr_loss_1 + pyr_loss_2 + pyr_loss_4 + hist_loss_1 + hist_loss_2 + hist_loss_4
            self.log('denoiser_train_loss', train_loss, prog_bar=True)
            
            return train_loss
            
        if optimizer_idx == 1 : # discriminator
            
            fake , _ , _  = self.denoiser(noisy)
            real = clean 
            
            
            B = fake.shape[0]

            real_bridge, real_x0, real_x2, real_x4 = self.discriminator(real)
            fake_bridge, fake_x0, fake_x2 , fake_x4 = self.discriminator(fake)
            
            
            real_ravel =  torch.concat([torch.reshape(real_bridge, (B,-1)),
                                torch.reshape(real_x0, (B,-1)), 
                                torch.reshape(real_x2, (B,-1)),
                                torch.reshape(real_x4, (B,-1))
                                ], axis=-1)
            
            real_loss = torch.mean(RELU(1.0 - real_ravel))
            
            fake_ravel =  torch.concat([torch.reshape(fake_bridge, (B,-1)),
                                torch.reshape(fake_x0, (B,-1)), 
                                torch.reshape(fake_x2, (B,-1)),
                                torch.reshape(fake_x4, (B,-1))
                                ], axis=-1)
            
            fake_loss = torch.mean(RELU(1.0 + fake_ravel))
            
            
            loss = (real_loss + fake_loss) / 2 
            
            self.log('disc_train_loss', loss, prog_bar = True)
            
            return loss

        
    def validation_step(self, batch, batch_idx, *args, **kwargs) :
    
        clean, noisy, _ = batch 
        
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)
        
        
            
        denoised, denoised_2, denoised_4 = self.denoiser(noisy)
        clean_2, clean_4 = self._rescale_gt_2d(clean)
        
        l1_loss_1 = Loss_L1(clean, denoised)
        l1_loss_2 = Loss_L1(clean_2, denoised_2)
        l1_loss_4 = Loss_L1(clean_4, denoised_4)
        
        pyr_loss_1 = Loss_PYR(clean, denoised, levels = 3)
        pyr_loss_2 = Loss_PYR(clean_2, denoised_2, levels = 3)
        pyr_loss_4 = Loss_PYR(clean_4, denoised_4, levels = 3)
        
        hist_loss_1  = Loss_Hist(clean, denoised)
        hist_loss_2  = Loss_Hist(clean_2, denoised_2)
        hist_loss_4  = Loss_Hist(clean_4, denoised_4)
        
        self.log('denoiser_val_l1_loss', l1_loss_1 + l1_loss_2 + l1_loss_4)
        self.log('denoiser_val_pyr_loss', pyr_loss_1 + pyr_loss_2 + pyr_loss_4)
        self.log('denoiser_val_hist_loss', hist_loss_1 + hist_loss_2 + hist_loss_4)
        
        val_loss = l1_loss_1 + l1_loss_2 + l1_loss_4 + pyr_loss_1 + pyr_loss_2 + pyr_loss_4 + hist_loss_1 + hist_loss_2 + hist_loss_4
        self.log('denoiser_val_loss', val_loss, prog_bar=True)
        
        return val_loss
            
            
    def configure_optimizers(self):
        opt_denoiser = torch.optim.Adam(
            self.denoiser.parameters(), lr = self.lr
        )
        
        opt_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr = self.lr
        )
        
        denoiser_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_denoiser, mode='min', patience = 5, threshold=0.001, verbose=True)
        
        discriminator_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_discriminator, mode='min', patience = 5, threshold=0.001, verbose=True)
        
        return [
            {
                'optimizer' : opt_denoiser, 
                'lr_scheduler' : {
                    'scheduler' : denoiser_lr_scheduler, 
                    'monitor'   : 'denoiser_val_loss', 
                    'frequency' : 1 
                }
            }, 
            {
                'optimizer' : opt_discriminator, 
                'lr_scheduler' : {
                    'scheduler' : discriminator_lr_scheduler, 
                    'monitor'   : 'disc_train_loss', 
                    'frequency' : 1 
                }
            }
            
        ]
    
    def on_train_epoch_end(self) :
        
        # visualising w and b 
        for name, params in self.named_parameters() : 
            self.logger.experiment.add_histogram(name, params, self.current_epoch)