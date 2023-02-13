import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

import torchvision

from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio

import pytorch_lightning as pl 



# generator 
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        #v1
        # self.model = nn.Sequential(
        #     *block(latent_dim, 128, normalize=False),
        #     *block(128, 256),
        #     *block(256, 512),
        #     *block(512, 1024),
        #     nn.Linear(1024, int(np.prod(img_shape))),
        #     nn.Tanh()
        # )

        #v2
        self.model = nn.Sequential(
        *block(latent_dim, 256, normalize=False),
        *block(256, 512, normalize=False),
        *block(512, 1024, normalize=False),
        nn.Linear(1024, int(np.prod(img_shape))),
        nn.Tanh()
        )


    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# discriminator
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        #v1

        # self.model = nn.Sequential(
        #     nn.Linear(int(np.prod(img_shape)), 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(128,1),
        #     nn.Sigmoid(),
        # )

        #v2 
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat) # real or fake classification

        return validity

# gan
class GAN(pl.LightningModule) : 
    def __init__(self, 
                channels, 
                width, 
                height,
                latent_dim : int = 100 , 
                lr : float = 2e-4, 
                b1 : float = 0.5, 
                b2 : float = 0.999, 
                batch_size : int  = 64,
                 **kwargs) :
        super().__init__()
        self.save_hyperparameters()

        img_shape = (channels, width, height)
        # add generator and discriminator networks
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape= img_shape)
        self.discriminator = Discriminator(img_shape= img_shape)

        # random noise for generating images 
        self.validation_z = torch.randn(8, self.hparams.latent_dim) # an array of 8 random noise images of latent dimensions

        # example input image to construct model summary ( using MOdel Summary Callback)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim) # DOUBT: why the 2 in dim=0 ?

    def forward(self, z) : 
        generated_imgs = self.generator(z)
        discriminator_output = self.discriminator(generated_imgs) # needed to print decoder I/o sizes in model summary 
        return generated_imgs 

    def adversarial_loss(self, y_hat, y) : 
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self,batch,batch_idx, optimizer_idx) : 
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim) # batch size x latent dimension
        z = z.type_as(imgs) # change the datatype and device of noise samples

        with torch.cuda.amp.autocast() : 
            # train the generator 
            if optimizer_idx == 0 : 

                # generate imgs 
                self.generated_imgs = self(z) # calls forward method 

                # log sampled images
                # rand_idx = torch.randint(0,self.hparams.batch_size,(6,)) # throws error on mps
                # rand_idx = rand_idx.type_as(imgs)
                # rand_idx = rand_idx.int()

                sample_imgs = self.generated_imgs[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('generated_images', grid, 0 ) # Add images generated at the start of generator training  

                # ground truth labels / indicates the images are fake
                valid = torch.ones(imgs.size(0), 1) # batch_size X 1 
                valid = valid.type_as(imgs)

                # adversarial loss : binary cross entropy
                g_loss = self.adversarial_loss(self.discriminator(self(z)), valid) # discriminator loss on the generated images.
                self.log('g_loss', g_loss, prog_bar=True)
                
                return g_loss

        with torch.cuda.amp.autocast() : 
            # train discriminator 
            if optimizer_idx == 1 : 
                # measure discriminator's ability to classify real from generated samples 

                # creating labels , how well can it label real ?
                valid = torch.ones(imgs.size(0), 1) # batch_size X 1 
                valid = valid.type_as(imgs)

                real_loss = self.adversarial_loss(self.discriminator(imgs), valid) 

                # create labels , how well can it label fake ?
                fake = torch.zeros(imgs.size(0),1)
                fake = fake.type_as(imgs)

                fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake) # D(G(z)) , detach : to ensure no backprop is performed.

                # discriminator loss is average of real_loss and fake_loss
                d_loss = (real_loss + fake_loss) / 2 
                self.log(
                    'd_loss', d_loss, prog_bar = True
                )

                return d_loss 

        
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr = lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr = lr, betas=(b1,b2))

        return [opt_g, opt_d], []

        # to do multiple passes per optimser 
        # return (
        #     {'optimizer': opt_g, 'frequency': 1},
        #     {'optimizer': opt_d, 'frequency': n_critic}
        # )

    def on_train_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)
        
        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'generated_images@{self.global_step}', grid, global_step= self.global_step)


        # The idea is psnr should be much higher than a random image. 
        # ssim with a random image should be low 

        # creating random images 
        random_imgs = torch.rand(sample_imgs.shape).type_as(sample_imgs).float()

        # psnr 
        g_psnr = peak_signal_noise_ratio(sample_imgs, random_imgs)
        self.log('g_psnr', g_psnr)

        # ssim 
        g_ssim = structural_similarity_index_measure(sample_imgs, random_imgs)
        self.log('g_ssim', g_ssim)

        # visualising w and b 
        for name, params in self.named_parameters() : 
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

            

        
