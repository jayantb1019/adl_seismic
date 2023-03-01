import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import pytorch_lightning as pl 


import torchmetrics
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure, multiscale_structural_similarity_index_measure 
from torchmetrics.classification import BinaryHingeLoss

import pdb 

import sys 
sys.path.append('../utils')

from adl_MODELS import Efficient_Unet_disc , Efficient_Unet
from adl_loss import Loss_L1 , Loss_PYR, Loss_Hist, HingeLoss, compute_gradient_penalty

#TODO : Change the milestones (list of epoch indices) in MultiStepLR schedulers.

# RELU = nn.ReLU(inplace=False)
# HINGE = BinaryHingeLoss().to('cuda')

# def HINGE(preds, target) : # tensor, tensor
#     return torch.mean(torch.max(torch.zeros_like(preds) , torch.ones_like(preds) - preds * target)).to('cuda')


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
        self.use_dropout = denoiser_config['use_dropout']
        
        self.model = Efficient_Unet(in_ch=1, out_ch = 1, filter_base=32, bias=False, use_dropout = self.use_dropout)

        # reflection padding for tests
        self.reflection_pad = nn.ReflectionPad2d(self.patch_size // 2)
        self.hflipper = torchvision.transforms.RandomHorizontalFlip(p=1)
        
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

        # pdb.set_trace()
        
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)
        
        denoised, denoised_2, denoised_4 = self(noisy)
        clean_2, clean_4 = self._rescale_gt_2d(clean)

        # clamp all model outputs to [-1,1]
        
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

        train_psnr = peak_signal_noise_ratio(denoised.detach(), clean.detach())
        train_ssim = structural_similarity_index_measure(denoised.detach(), clean.detach(), sigma=0.5, kernel_size = 5, )
        # train_msssim = multiscale_structural_similarity_index_measure(denoised.detach(), clean.detach(), sigma=0.5, data_range=2.0)

        self.log('train_psnr', train_psnr)
        self.log('train_ssim', train_ssim)
        # self.log('train_msssim', train_msssim)
        
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

        val_psnr = peak_signal_noise_ratio(denoised.detach(), clean.detach())
        val_ssim = structural_similarity_index_measure(denoised.detach(), clean.detach(), sigma=0.5, kernel_size = 5, )
        # val_msssim = multiscale_structural_similarity_index_measure(denoised.detach(), clean.detach(), sigma=0.5, data_range=2.0)

        self.log('val_psnr', val_psnr)
        self.log('val_ssim', val_ssim)
        # self.log('val_msssim', val_msssim)

        return val_loss 
    
    def test_step(self, batch, batch_idx, *args, **kwargs) : 

        clean, noisy, _ = batch 
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)

        noisy_refpad = self.reflection_pad(noisy) # perform inference on a padded patch 

        denoised, denoised_2, denoised_4 = self.model(noisy_refpad)
        
        denoised = torch.clamp(denoised, -1,1) # Just a precaution

        denoised = denoised[:,:, self.patch_size //2 : self.patch_size //2 + self.patch_size , self.patch_size //2 : self.patch_size //2 + self.patch_size ] # recover original patch

        

        test_psnr = peak_signal_noise_ratio(denoised.detach(), clean.detach())
        test_ssim = structural_similarity_index_measure(denoised.detach(), clean.detach(), sigma=0.5, kernel_size = 5, )

        self.log('test_psnr', test_psnr)
        self.log('test_ssim', test_ssim)


    def predict_step(self, batch, batch_idx, *args, **kwargs) : 

        clean, noisy, _ = batch 
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)

        # test time augmentations
        noisy_pr = -1 * noisy
        noisy_flipped = self.hflipper(noisy)
        noisy_pr_flipped = -1 * noisy_flipped






    # print(test_psnr, test_ssim)   
    
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
        
        self.model = model.eval() # trained denoiser model
        self.hinge_loss = HingeLoss()
        
        # labels
        self.real_label = 1 
        self.fake_label = -1 
        self.gen_label = self.fake_label
        
        
        self.disc_model = Efficient_Unet_disc(in_ch=1, out_ch=1, negative_slope = self.negative_slope, filter_base = 8 , bias=False)
        
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
        '''
            The discriminator should output a patch of 1s for clean input, a patch of 0s for denoised input. 
            Hence, loss for clean input is 1 - sigmoid(disc output for clean) , loss for pred output is ( sigmoid(disc output for pred))
        '''
        # loss for true sample
        clean, noisy, _ = batch 
        
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)
        
        
        gt_bridge, gt_x0, gt_x2, gt_x4 = self.disc_model(clean) # classifications ( 1 x 1 conv outputs)
        B = clean.shape[0]
        
        true_ravel = torch.concat([
                                torch.reshape(gt_bridge, (B,-1)),
                                torch.reshape(gt_x0, (B,-1)), 
                                torch.reshape(gt_x2, (B,-1)),
                                torch.reshape(gt_x4, (B,-1))
                                ], axis=-1)
        # loss_true = torch.mean(RELU(1.0 - true_ravel)) + torch.mean(RELU(1.0 - torch.reshape(gt_bridge, (B,-1))))# loss ( zero if network output = 1 , else positive value.)
        # loss_true = nn.HingeEmbeddingLoss(true_ravel, torch.ones_like(true_ravel))

        # one sided label smoothing , according to https://arxiv.org/abs/1701.00160
        real_label = (self.real_label * 0.1 * torch.randint(9,13,(1,))).type_as(true_ravel)

        loss_true = self.hinge_loss(true_ravel, real_label * torch.ones_like(true_ravel))
        
    
        y = noisy
        y_bridge, y_pred, y_pred_x2, y_pred_x4 = self.disc_model(self.model(y)[0]) # denoised
        B = y.shape[0]

        # Compute the loss for the denoised sample
        pred_ravel = torch.concat([
                            torch.reshape(y_bridge, (B,-1)),
                            torch.reshape(y_pred, (B,-1)), 
                            torch.reshape(y_pred_x2, (B,-1)),
                            torch.reshape(y_pred_x4, (B,-1))
                            ], axis=-1)
        
        # loss_pred = torch.mean(RELU(1.0 + (pred_ravel)))  + torch.mean(RELU(1.0 + (torch.reshape(y_bridge, (B,-1))))) # loss ( zero if network output = 0 , else positive value)

        # loss_pred = torch.mean(0.0 - (pred_ravel)) + torch.mean(0.0 - (torch.reshape(y_bridge,(B,-1)))) # isn't this correct ? since, we want the predicted loss to be close to zero ?
        # loss_pred = HINGE(pred_ravel, -1 * torch.ones_like(pred_ravel))
        loss_pred = self.hinge_loss(pred_ravel, self.fake_label * torch.ones_like(pred_ravel))
        
        loss = (loss_true + loss_pred) /2
        
        self.log('disc_train_loss', loss, prog_bar=True )
        self.log('disc_train_true_loss', loss_true)
        self.log('disc_train_fake_loss', loss_pred)
        
        return loss
    
    def validation_step(self, batch, batch_idx, *args, **kwargs) : ## look at this again!! discriminator should train on half a batch of clean and half a batch of denoised
        
        # loss for true sample
        clean, noisy, _ = batch 
        
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)
        
        
        gt_bridge, gt_x0, gt_x2, gt_x4 = self.disc_model(clean)
        B = clean.shape[0]
        
        true_ravel = torch.concat([
                            torch.reshape(gt_bridge, (B,-1)),
                                torch.reshape(gt_x0, (B,-1)), 
                                torch.reshape(gt_x2, (B,-1)),
                                torch.reshape(gt_x4, (B,-1))
                                ], axis=-1)
    

        # loss_true = torch.mean(RELU(1.0 - (true_ravel)))  + torch.mean(RELU(1.0 - (torch.reshape(gt_bridge, (B,-1)))))

        # one sided label smoothing , according to https://arxiv.org/abs/1701.00160
        real_label = (self.real_label * 0.1 * torch.randint(9,13,(1,))).type_as(true_ravel)

        loss_true = self.hinge_loss(true_ravel, real_label * torch.ones_like(true_ravel))
    
        y = noisy
        y_bridge, y_pred, y_pred_x2, y_pred_x4 = self.disc_model(self.model(y)[0])
        B = y.shape[0]

        # Compute the loss for the predicted sample
        pred_ravel = torch.concat([
                            torch.reshape(y_bridge, (B,-1)),
                            torch.reshape(y_pred, (B,-1)), 
                            torch.reshape(y_pred_x2, (B,-1)),
                            torch.reshape(y_pred_x4, (B,-1))
                            ], axis=-1)

        # loss_pred = torch.mean(RELU(1.0 + (pred_ravel)))  + torch.mean(RELU(1.0 + (torch.reshape(y_bridge, (B,-1)))))
        loss_pred = self.hinge_loss(pred_ravel, self.fake_label * torch.ones_like(pred_ravel))
        
        loss = (loss_true + loss_pred) /2
        
        self.log('disc_val_loss', loss, prog_bar=True )
        self.log('disc_val_true_loss', loss_true)
        self.log('disc_val_fake_loss', loss_pred)
        
        return loss
    





    def configure_optimizers(self):
        
        lr = self.lr

        optimiser = torch.optim.Adam(self.disc_model.parameters(), lr = lr , betas=(0.5,0.999))
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
        self.hinge_loss = HingeLoss()
        
        data_config = config['train']['data']
        self.patch_size = data_config['patch_size']
        self.batch_size = data_config['batch_size']
        
        adl_config = config['train']['ADL']
        self.denoiser_lr = adl_config['denoiser_lr']
        self.discriminator_lr = adl_config['discriminator_lr']
        self.optimizer = adl_config['optimizer']
        self.lr_scheduler = adl_config['lr_scheduler']['type']
        self.gamma = adl_config['lr_scheduler']['kwargs']['gamma']
        
        self.lambda1 = adl_config['lambda1']
        
        print('gan lambda : ', self.lambda1)
        
        # labels 
        self.real_label = 1
        self.fake_label = -1
        self.gen_label = self.fake_label

        # reflection padding for tests
        self.reflection_pad = nn.ReflectionPad2d(self.patch_size // 2)
        
        self.save_hyperparameters(ignore=['denoiser', 'discriminator', 'hinge_loss'])
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
            
            # gan loss
            fake_bridge, fake_x0, fake_x2, fake_x4 = self.discriminator(denoised)
            B = noisy.shape[0]
            fake_ravel =  torch.concat([
                                torch.reshape(fake_bridge, (B,-1)),
                                torch.reshape(fake_x0, (B,-1)), 
                                torch.reshape(fake_x2, (B,-1)),
                                torch.reshape(fake_x4, (B,-1))
                                ], axis=-1)
            
            # fake_loss = torch.mean(RELU(1.0 - fake_ravel))
            
            # fake_loss = torch.mean(RELU(1.0 - (fake_ravel))) 
            # fake_loss = -1 * HINGE(fake_ravel, torch.ones_like(fake_ravel))
            
            # fake_loss = -1 * torch.mean(fake_ravel)
            
            fake_loss = self.hinge_loss(fake_ravel, self.gen_label * torch.ones_like(fake_ravel)) # based on geometric gan hinge loss 
            
            # model loss
            
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
            self.log('denoiser_gan_loss', fake_loss)
            
            train_loss = 1 * (l1_loss_1 + l1_loss_2 + l1_loss_4 + pyr_loss_1 + pyr_loss_2 + pyr_loss_4 + hist_loss_1 + hist_loss_2 + hist_loss_4) + self.lambda1 * fake_loss
    
            
            
            self.log('denoiser_train_loss', train_loss, prog_bar=True)
            
            # pdb.set_trace()
            
            return train_loss
            
        if optimizer_idx == 1 : # discriminator
            
            fake , _ , _  = self.denoiser(noisy)
            real = clean 
            
            
            B = fake.shape[0]

            real_bridge, real_x0, real_x2, real_x4 = self.discriminator(real)
            fake_bridge, fake_x0, fake_x2 , fake_x4 = self.discriminator(fake)
            
            
            real_ravel =  torch.concat([
                                torch.reshape(real_bridge, (B,-1)),
                                torch.reshape(real_x0, (B,-1)), 
                                torch.reshape(real_x2, (B,-1)),
                                torch.reshape(real_x4, (B,-1))
                                ], axis=-1)
            
            # real_loss = torch.mean(RELU(1.0 - (real_ravel))) + torch.mean(RELU(1.0 - (torch.reshape(real_bridge, (B,-1)))))
            # real_loss = HINGE(real_ravel, torch.ones_like(real_ravel))
            # real_loss = torch.max(torch.zeros_like(real_ravel), torch.ones_like(real_ravel) - torch.mean(real_ravel))

            # one sided label smoothing , according to https://arxiv.org/abs/1701.00160
            real_label = (self.real_label * 0.1 * torch.randint(9,13,(1,))).type_as(real_ravel)
            
            real_loss = self.hinge_loss(real_ravel, real_label * torch.ones_like(real_ravel))
            
            fake_ravel =  torch.concat([
                                torch.reshape(fake_bridge, (B,-1)),
                                torch.reshape(fake_x0, (B,-1)), 
                                torch.reshape(fake_x2, (B,-1)),
                                torch.reshape(fake_x4, (B,-1))
                                ], axis=-1)
            
            # fake_loss = torch.mean(RELU(1.0 + fake_ravel)) + torch.mean(RELU(1.0 + (torch.reshape(fake_bridge, (B,-1)))))
            
            # fake_loss = HINGE(fake_ravel, -1 * torch.ones_like(fake_ravel))
            # fake_loss = torch.max(torch.zeros_like(fake_ravel), torch.ones_like(fake_ravel) + torch.mean(fake_ravel))
            
            fake_loss = self.hinge_loss(fake_ravel, self.fake_label * torch.ones_like(fake_ravel))
            
            loss = (real_loss + fake_loss) /2
            # + 10 * compute_gradient_penalty(self.discriminator, real_ravel, fake_ravel) # lambda gp = 10
            
            
            self.log('disc_train_loss', loss, prog_bar = True)
            self.log('disc_real_loss', real_loss)
            self.log('disc_fake_loss', fake_loss)
            # pdb.set_trace()
            
            return loss

        
    def validation_step(self, batch, batch_idx, *args, **kwargs) :
    
        clean, noisy, _ = batch 
        
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)
        
        
            
        denoised, denoised_2, denoised_4 = self.denoiser(noisy)
        clean_2, clean_4 = self._rescale_gt_2d(clean)
        
        # gan loss
        fake_bridge, fake_x0, fake_x2, fake_x4 = self.discriminator(denoised)
        B = noisy.shape[0]
        fake_ravel =  torch.concat([ torch.reshape(fake_bridge, (B,-1)),
                            torch.reshape(fake_x0, (B,-1)), 
                            torch.reshape(fake_x2, (B,-1)),
                            torch.reshape(fake_x4, (B,-1))
                            ], axis=-1)
        
        # fake_loss = torch.mean(RELU(1.0 - fake_ravel))
        
        # fake_loss = torch.mean(RELU(1.0 - (fake_ravel))) + torch.mean(RELU(1.0 - (torch.reshape(fake_bridge, (B,-1)))))
        # fake_loss = -1 * HINGE(fake_ravel, torch.ones_like(fake_ravel)) 
        # fake_loss = -1 * torch.mean(fake_ravel)
        fake_loss = self.hinge_loss(fake_ravel, self.gen_label * torch.ones_like(fake_ravel))

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
        self.log('denoiser_val_gan_loss', fake_loss)
        
        val_loss =  (l1_loss_1 + l1_loss_2 + l1_loss_4 + pyr_loss_1 + pyr_loss_2 + pyr_loss_4 + hist_loss_1 + hist_loss_2 + hist_loss_4 )+  self.lambda1 * fake_loss   
        # val_loss = fake_loss 

        self.log('denoiser_val_loss', val_loss, prog_bar=True)

        val_psnr = peak_signal_noise_ratio(denoised.detach(), clean.detach())
        val_ssim = structural_similarity_index_measure(denoised.detach(), clean.detach(), sigma=0.5, kernel_size = 5, )
        # val_msssim = multiscale_structural_similarity_index_measure(denoised.detach(), clean.detach(), sigma=0.5, data_range=2.0)

        self.log('val_psnr', val_psnr)
        self.log('val_ssim', val_ssim)
        
        return val_loss
            
    
    def test_step(self, batch, batch_idx, *args, **kwargs) : 
        clean, noisy, _ = batch 
        clean = clean.to(torch.float32).to(self.device)
        noisy = noisy.to(torch.float32).to(self.device)

        noisy_refpad = self.reflection_pad(noisy) # perform inference on a padded patch 

        denoised, denoised_2, denoised_4 = self.denoiser(noisy_refpad)
        
        denoised = torch.clamp(denoised, -1,1) # Just a precaution

        denoised = denoised[:,:,self.patch_size //2 : self.patch_size //2 + self.patch_size , self.patch_size //2 : self.patch_size //2 + self.patch_size ] # recover original patch

        test_psnr = peak_signal_noise_ratio(denoised.detach(), clean.detach()) # let's not give data range
        test_ssim = structural_similarity_index_measure(denoised.detach(), clean.detach(), sigma=0.5, kernel_size = 5, )

        self.log('test_psnr', test_psnr)
        self.log('test_ssim', test_ssim)

        # print(test_psnr, test_ssim)


    def configure_optimizers(self):
        print('adl denoiser lr :', self.denoiser_lr)
        print('adl discrminator lr :', self.discriminator_lr)
        opt_denoiser = torch.optim.Adam(
            self.denoiser.parameters(), lr = self.denoiser_lr , betas=(0.5,0.999) # according to https://arxiv.org/abs/1511.06434
        )
        
        opt_discriminator = torch.optim.Adam(
            self.discriminator.parameters(), lr = self.discriminator_lr , betas=(0.5,0.999)
        )
        
        denoiser_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_denoiser, mode='min', patience = 5, threshold=0.001, verbose=True)
        
        discriminator_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_discriminator, mode='min', patience = 5, threshold=0.001, verbose=True)
        
        return [
            {
                'optimizer' : opt_denoiser, 'frequency' : 5 , 
                'lr_scheduler' : {
                    'scheduler' : denoiser_lr_scheduler, 
                    'monitor'   : 'denoiser_val_loss', 
                    'frequency' : 1
                }
            }, 
            {
                'optimizer' : opt_discriminator, 'frequency' : 1,
                # 'lr_scheduler' : {
                #     'scheduler' : discriminator_lr_scheduler, 
                #     'monitor'   : 'disc_train_loss', 
                #     'frequency' : 11 
                # }
            }
            
        ]
    
    def on_train_epoch_end(self) :
        
        # visualising w and b 
        for name, params in self.named_parameters() : 
            self.logger.experiment.add_histogram(name, params, self.current_epoch)