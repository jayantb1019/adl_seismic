import numpy as np 
import pandas as pd
import os
import sys 


from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl 
import multiprocessing

import albumentations as A

sys.path.append('../utils/')
from data import patch_index_generation, select_patch
from augmentations import polarity_reversal, random_trace_dropout, random_high_noise_trace, random_amp_attenuation, random_trace_shuffle, horizontal_flip, random_amp_shift, rotate_patch

from skimage.util import random_noise
from skimage.filters import butterworth

class FaciesMarkDataset(Dataset) : 
    def __init__(self, dataset, labels, patch_info, training_config , mode = 'train') :
        super().__init__()
        self.patch_size = training_config['patch_size']
        self.stride = training_config['stride']
        self.data_mode = training_config['data_mode']        
        self.dataset = dataset
        self.labels = labels
        self.patch_info = patch_info # dataframe
        
        self.noise_mode = training_config['noise_mode']
        self.noise_factor  = training_config['noise_factor']

        self.global_seed = training_config['seed']
        
        self.mode = mode # could be train, val, test 
        
        self.augmentations = training_config['augmentations']
        
        print('Augmentations', self.augmentations)

        # Data Augmentation transforms. NOTE : A.Compose is not working with multiple A.Lambda, hence they are split into multiple transform sets
        
        self.tset_1 = A.Lambda(name='polarity_reversal', image=polarity_reversal, p = 0.2 )
        self.tset_2 =  A.OneOf([
                            A.Lambda(name='horizontal_flip', image = horizontal_flip, p = 0.2 ),
                        A.Lambda(name='rotate_patch', image = rotate_patch, p = 0.2 ),
                        ])
        self.tset_3 =  A.OneOf([
                            A.Lambda(name='random_trace_dropout', image = random_trace_dropout, p = 0.2 ),
                            A.Lambda(name='random_high_noise_trace', image = random_high_noise_trace, p = 0.2 ),
                            A.Lambda(name='random_trace_shuffle', image = random_trace_shuffle, p = 0.2 ),
                        ])

        self.tset_4 = A.OneOf([
                            A.Lambda(name='random_amp_attenuation', image = random_amp_attenuation , p = 0.2 ), 
                            A.Lambda(name='random_amp_shift', image = random_amp_shift , p = 0.2 ), 
                        ])


        

    def __len__(self) : 
        return self.patch_info.shape[0]

    def __getitem__(self, index) :
        shape = [self.patch_size] * 3 if self.data_mode == '3d' else [self.patch_size] * 2 
        data , noisy_data, label = np.zeros(shape, dtype=np.float32) , np.zeros(shape, dtype=np.float32),  np.zeros(shape, dtype=np.float32)
        
        data_ = select_patch(self.dataset, self.patch_info, index, self.patch_size, self.data_mode)
        label_ = select_patch(self.labels, self.patch_info, index, self.patch_size, self.data_mode)
        
        # transpose to bring twt as the middle dimension 
        
        if self.data_mode == '3d' : 
            data_ = np.transpose(data_, [0,2,1])
            label_ = np.transpose(label_, [0,2,1])
        else : 
            data_ = data_.T
            label_ = label_.T
        
        noisy_data_ = np.zeros_like(data_)
        
        if self.noise_mode == 'gaussian' : 
            noisy_data_ = random_noise(data_, 'gaussian', seed = self.global_seed , clip=True, mean = 0 , var = self.noise_factor)
        if self.noise_mode == 'poisson' : 
            noisy_data_ = random_noise(data_, 'gaussian', seed = self.global_seed , clip=True, mean = 0.05, var = 0.05)
            
        if self.noise_mode == 'mixed' : # approximate poisson as gaussian
            noisy_data_ = np.clip(random_noise(data_, 'gaussian', seed = self.global_seed , clip=True, mean = 0.05, var = 0.05) + \
                         random_noise(data_, 'gaussian', seed = self.global_seed , clip=True , mean= 0, var = 0.05 ),-1,1)
            
        if self.noise_mode == 'lpf' : 
            noisy_data_ = butterworth(data_, cutoff_frequency_ratio=0.42, high_pass=False, order=5.0, channel_axis=None) # 50 hz 
        
        data_shape = data_.shape
        
        
        # add augmentations for training and val
        if self.augmentations : 
            if (self.mode == 'train') or (self.mode == 'val') : 
                data_stack = np.stack([data_, noisy_data_], axis=0)
                aug_1 = self.tset_1(image=data_stack )['image'] 
                aug_2 = self.tset_2(image = np.stack(aug_1, axis=0))['image']
                aug_3 = self.tset_3(image = np.stack(aug_2, axis=0))['image']
                aug_4 = self.tset_4(image = np.stack(aug_3, axis=0))['image']

                data_, noisy_data_ = aug_4
        
        # ensure shape consistency even in edge patches
        
        if self.data_mode == '3d' : 
            data[ : data_shape[0], :data_shape[1], :data_shape[2]] = data_
            noisy_data[ : data_shape[0], :data_shape[1], :data_shape[2]] = noisy_data_
            label[ : data_shape[0], :data_shape[1], :data_shape[2]] = label_
        else : 
            data[ : data_shape[0], : data_shape[1]] = data_
            noisy_data[ : data_shape[0], : data_shape[1]] = noisy_data_
            label[ : data_shape[0], : data_shape[1]] = label_
        
        return np.expand_dims(data, axis=0) , np.expand_dims(noisy_data, axis=0), np.expand_dims(label, axis=0) 
       
class FaciesMarkDataModule(pl.LightningDataModule) : 
    def __init__(self, training_config) : 
        super().__init__() 
        self.training_config = training_config
        
        self.data_mode  = training_config['data_mode']
        self.patch_size = training_config['patch_size']
        self.stride = training_config['stride']
        self.batch_size = training_config['batch_size']
        self.noise_mode = training_config['noise_mode']
        self.noise_factor = training_config['noise_factor']
        
        self.train_data_path = os.path.join(training_config['dir']['data_root'],training_config['dir']['train_data_path'])
        self.val_data_path = os.path.join(training_config['dir']['data_root'],training_config['dir']['val_data_path'])
        self.test_data_path = os.path.join(training_config['dir']['data_root'],training_config['dir']['test_data_path'])
        self.train_labels_path = os.path.join(training_config['dir']['data_root'],training_config['dir']['train_labels_path'])
        self.val_labels_path  = os.path.join(training_config['dir']['data_root'],training_config['dir']['val_labels_path'])
        self.test_labels_path = os.path.join(training_config['dir']['data_root'],training_config['dir']['test_labels_path'])
        
        self.train_patch_index_path = os.path.join(training_config['dir']['data_root'], f"train_patches_{self.patch_size}_{self.stride}_{self.data_mode}.csv")
        self.val_patch_index_path = os.path.join(training_config['dir']['data_root'],f"test1_patches_{self.patch_size}_{self.stride}_{self.data_mode}.csv")
        self.test_patch_index_path = os.path.join(training_config['dir']['data_root'],f"test2_patches_{self.patch_size}_{self.stride}_{self.data_mode}.csv")
        
        
        self.cpu_count = multiprocessing.cpu_count()//2
        
    def prepare_data(self) : 
        patch_index_generation(self.train_data_path, self.stride, self.train_patch_index_path, self.data_mode)
        patch_index_generation(self.val_data_path, self.stride, self.val_patch_index_path, self.data_mode)
        patch_index_generation(self.test_data_path, self.stride, self.test_patch_index_path, self.data_mode)
    
    def setup(self, stage) :
        if stage == 'fit' or None : 
            self.train_dataset = FaciesMarkDataset(dataset=np.load(self.train_data_path), labels = np.load(self.train_labels_path) ,
                                                   patch_info= pd.read_csv(self.train_patch_index_path) ,  
                                                   training_config= self.training_config, mode='train')
            
            self.val_dataset = FaciesMarkDataset(dataset=np.load(self.val_data_path), labels = np.load(self.val_labels_path) ,
                                                   patch_info= pd.read_csv(self.val_patch_index_path) ,  
                                                   training_config= self.training_config, mode='val')
            
        if stage == 'test' : 
            self.test_dataset = FaciesMarkDataset(dataset=np.load(self.test_data_path), labels = np.load(self.test_labels_path) ,
                                                   patch_info= pd.read_csv(self.test_patch_index_path) ,  
                                                   training_config= self.training_config, mode='test')
    
    def train_dataloader(self) : 
        return DataLoader(self.train_dataset, shuffle=True,batch_size= self.batch_size, num_workers= self.cpu_count)
    
    def test_dataloader(self) : 
        return DataLoader(self.test_dataset, shuffle=False, batch_size= self.batch_size, num_workers= self.cpu_count)
    
    def val_dataloader(self) : 
        return DataLoader(self.val_dataset, shuffle=False,batch_size= self.batch_size, num_workers= self.cpu_count)