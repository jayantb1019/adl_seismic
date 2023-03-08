# Ref : Fang 2022 , BSNet : An unsupervised Blindspot Network for seismic data random noise attenuation

import os
import pdb
import torch 
from torch import nn
from torch.nn import functional, Sequential
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import datasets, transforms 

import torchmetrics
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, callbacks, profiler 
from pytorch_lightning.profiler import AdvancedProfiler


patch_size = 40
learning_rate = 1e-3
max_epochs=30
optimizer = 'adam'

# optimiser parameters
m_o = 0
v_o = 0
beta_1 = 0.9
beta_2 = 0.999
epsilon = 10e-8

weight_decay = 10e-6



class DIP(pl.LightningModule) : 
    def __init__(self, learning_rate=1e-4, image_size=256, *kargs, **kwargs) : 
        super().__init__() 
        
        self.ds1 = self.basic_downsampling_block(1,8)
        self.ds2 = self.basic_downsampling_block(8,16)
        self.ds3 = self.basic_downsampling_block(16,32,maxpool=False)
        self.ds4 = self.basic_downsampling_block(32,64,maxpool=False)
        self.ds5 = self.basic_downsampling_block(64,128,maxpool=False)

        self.us1 = self.basic_upsampling_block(128,64, bilinear=False)
        self.us2 = self.basic_upsampling_block(64,32, bilinear=False)
        self.us3 = self.basic_upsampling_block(32,16,bilinear=False)
        self.us4 = self.basic_upsampling_block(16,8)

        # self.maxpool1 = nn.MaxPool2d(kernel_size=3)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=3)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=3)

        self.upsamplingBL1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsamplingBL2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsamplingBL3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsamplingBL4 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = nn.Conv2d(kernel_size=3, in_channels=256, out_channels=256, padding='same')
        self.conv2 = nn.Conv2d(kernel_size=3,in_channels = 128, out_channels=128, padding='same')
        self.conv3 = nn.Conv2d(kernel_size=3,in_channels = 64, out_channels=64, padding='same')

        self.conv_last = nn.Conv2d(kernel_size=1,in_channels=16,out_channels=1, padding='same')

        self.learning_rate = learning_rate
        self.example_input_array = torch.Tensor(batch_size,1,patch_size, patch_size)
    
        # saving hyper parameters
    
        self.save_hyperparameters()

    def forward(self,x) : 

        # downsampling 

        x = self.ds1(x) # out features = 16
        x = self.ds2(x) # out_features = 32
        x = self.ds3(x) # out_features = 64

        sk3 = x
        x = self.maxpool1(x) 
        x = self.ds4(x) # out_features = 128

        sk4 = x 
        x = self.maxpool2(x)
        x = self.ds5(x) # out_features = 256

        sk5 = x 
        x = self.maxpool3(x)

        

        # upsampling 

        x = self.upsamplingBL1(x)
        x = self.conv1(x)
        x = torch.cat([x,sk5], 1) # concatenating features
        x = self.us1(x) # skip connection between ds5 and us1 , # out_features = 16

        x = self.upsamplingBL2(x)
        x = self.conv2(x)
        x = torch.cat([x,sk4], 1) # concatenating features
        x = self.us2(x) # skip connection between ds4 and us4, # out_features = 32

        x = self.upsamplingBL3(x)
        x = self.conv3(x)
        x = torch.cat([x,sk3], 1) # concatenating features
        x = self.us3(x) # skip connection between ds4 and us4, # out_features = 64    

        x = self.us4(x) # out_features = 128

        x = self.upsamplingBL4(x)

        return self.conv_last(x) # 1x1 convolution

    def basic_downsampling_block(self, in_channels, out_channels, maxpool=True) : 

        layers = [
                nn.Conv2d(kernel_size=3, in_channels = in_channels, out_channels=out_channels , padding='same' ),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(kernel_size=3, in_channels = out_channels, out_channels=out_channels  , padding='same' ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
        ]

        return nn.Sequential(*layers)

    def basic_upsampling_block(self, in_channels, out_channels, bilinear = True) : 
        layers = []
        if bilinear : 
            layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
        
        layers.extend([
                nn.Conv2d(kernel_size=3, in_channels = in_channels, out_channels=out_channels , padding='same'  ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(kernel_size=3, in_channels = out_channels, out_channels=out_channels , padding='same'  ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
        ])

        return nn.Sequential(*layers)

    def training_step(self,batch, batch_idx) : 
        x,x_c, m, m_c = batch # masked patch, masked patch complement, mask, mask_complement
        predicted = self.forward(x)
        masked_predicted_c = m_c * predicted
        loss = nn.functional.mse_loss(masked_predicted_c, x_c)
        self.log('train_loss', loss, prog_bar = True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) : 
        x,x_c, m, m_c = batch # masked patch, masked patch complement, mask, mask_complement

        predicted = self.forward(x)
        masked_predicted_c = m_c * predicted
        loss = nn.functional.mse_loss(masked_predicted_c, x_c)
        self.log('val_loss', loss, prog_bar = True, on_epoch=True)

        psnr = peak_signal_noise_ratio(masked_predicted_c, x_c)
        ssim = structural_similarity_index_measure(masked_predicted_c, x_c)
        

        self.log('val_psnr', psnr)
        self.log('val_ssim', ssim)

        return loss

    def configure_optimizers(self) : 
        lr = self.learning_rate
        optimiser = torch.optim.Adam(self.parameters(), lr = lr,betas=[0.9,0.999], eps=1e-7, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size = 15, gamma=0.1)
        return {
            'optimizer' : optimiser, 
            'lr_scheduler' : {
                'scheduler' : lr_scheduler, 
                'monitor'   : 'val_loss',
                'interval'  : 'epoch',
                'frequency' : 15
            }
        }

    def on_train_epoch_end(self) : 
        for name, params in self.named_parameters(): 
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
   
