import os 
import gc 

import sys
import json
import pdb
from rich.progress import Progress
from rich import inspect
import multiprocessing
import h5py

import numpy as np
import pandas as pd
import pytorch_lightning as pl 

import torch
from torch.utils.data import DataLoader, IterableDataset, random_split
from torchvision.transforms import ToTensor

from segysak import open_seisnc

# import albumentations as A # TODO : Add augmentations 

from albumentations.pytorch import ToTensorV2

sys.path.append('../utils')
from segy_input import prepare_data_xarray, create_patch_index,derive_n_samples
from segy_input import get_config, get_seisnc, get_patch_info,get_global_stats,calculate_global_stats

sys.path.append('../utils')
from random_noise import add_noise


class SEGYDataset(IterableDataset) : 

    def __init__(self, segy_filename, config_path,  mode, limit_samples = None, noise_factor = None, **kwargs) : 

        '''
            noise_factor : if a noise factor between 0 and 1 is passed, the network will noise the samples based on that , else mixed noise is used
            limit samples : pass a fraction between 0 and 1 to limit samples during train mode
        '''

        super().__init__()

        self.segy_filename = segy_filename # used for patch index, global_stats etc 

        self.config_path = config_path
        self.config = get_config(config_path)
        self.patch_size = self.config['patch_based_training']['patch_size']
        self.mode = mode # train, test, val

        self.limit_samples = limit_samples

        self.noise_factor = noise_factor 


        self.n_train_samples, self.n_val_samples , self.n_test_samples = derive_n_samples(self.segy_filename, self.config_path)

        if self.limit_samples : # override the number of train samples
            self.n_train_samples = self.limit_samples * self.n_train_samples

        if self.noise_factor : # override with provided noise factor
            self.noise_factors = [self.noise_factor]
        else : 
            self.noise_factors = self.config['patch_based_training']['noise_factors'] # all noise factors defined in config 

        self.mean, self.std, self.abs_max = get_global_stats(self.segy_filename, self.config_path)

        


    def per_worker_init(self, n_workers =1 ) : 
        # print("worker initialised")

        self.data_array = get_seisnc(self.segy_filename, self.config_path)

        # converting abs_max to float32
        self.abs_max = self.abs_max.astype(np.float32, casting='safe')

        self.train_patch_index_info , self.val_patch_index_info, self.test_patch_index_info = get_patch_info(self.segy_filename, self.config_path)

        if self.mode == 'train' : 
            self.n_samples_per_epoch_per_worker =  self.n_train_samples // n_workers

        if self.mode == 'val' : 
            self.n_samples_per_epoch_per_worker =  self.n_val_samples // n_workers

        if self.mode == 'test' : 
            self.n_samples_per_epoch_per_worker =  self.n_test_samples // n_workers


        # each worker must have a different seed , else all worker return the same data 
        seed = torch.initial_seed()

        self.rng = np.random.default_rng(seed=seed) # random number generator

    def __iter__(self) : 

        # print("Data loading started with ", self.n_samples_per_epoch_per_worker, ' samples per epoch per worker')

        for _ in range(self.n_samples_per_epoch_per_worker) : 
            
            patch_idx = self.rng.integers(low=0, high = self.n_samples_per_epoch_per_worker)

            # print('patch index ' , patch_idx)

            # get geometry of a patch at patch_idx

            if self.mode == 'train' : 
                iline, xline_start, twt_start = self.train_patch_index_info.iloc[patch_idx].to_dict().values()

            if self.mode == 'test' : 
                iline, xline_start, twt_start = self.test_patch_index_info.iloc[patch_idx].to_dict().values()

            if self.mode == 'val' : 
                iline, xline_start, twt_start = self.val_patch_index_info.iloc[patch_idx].to_dict().values()



            iline_idx , xline_start_idx, twt_start_idx = get_indices(self.data_array,int(iline), int(xline_start), twt_start )

            xline_end_idx , twt_end_idx = xline_start_idx + self.patch_size , twt_start_idx + self.patch_size 

            patch = self.data_array.isel(iline=iline_idx, xline = slice(xline_start_idx, xline_end_idx), twt = slice(twt_start_idx, twt_end_idx))
            
            # print('Loading patch')
           
            # convert to numpy array and transpose

            patch = patch.compute(scheduler='single-threaded').data.values.T # Reason for using single-threaded Ref to https://github.com/dask/dask/issues/6664
  
            # convert to float32 (seisnc is creating float64 by default because of twt index's dtype = float64. The data values are originally float32 only )

            # print('patch dtype', patch.dtype)

            patch = patch.astype(np.float32, casting='unsafe')

            

            # clip and normalise by abs_max
            patch = np.clip(patch,a_min = -1 * self.abs_max, a_max = self.abs_max)
            patch = patch / self.abs_max
            

            # if size of patch is less than config patch size pad with zeros 
            mask = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            mask[:patch.shape[0], :patch.shape[1]] = patch 

            patch = mask

            # add channel dimension
            patch = np.expand_dims(patch, axis=0)

            
            # pick a random noise level 
            random_noise_factor = np.random.choice(self.noise_factors)


            noisy_patch = add_noise(patch, 'gaussian', 0, random_noise_factor)

            noise = noisy_patch - patch

            yield [noisy_patch, patch, noise]




def worker_init_fn(worker_id) : 

    '''
    Configures each dataset worker process. it calls SEGYDataset.per_worker_init().
    '''   

    worker_info = torch.utils.data.get_worker_info()
    # print(worker_info.dataset, worker_info)

    if worker_info is None : 
        print('No worker info')
    else : 
        
        dataset_obj = worker_info.dataset # the subset copy of worker process
        dataset_obj.per_worker_init(n_workers = worker_info.num_workers) 



def get_indices(seisnc, iline, xline, twt) : 
    return np.where(seisnc['iline'].values == iline)[0][0] , np.where(seisnc['xline'].values == xline)[0][0], np.where(seisnc['twt'].values == twt)[0][0]


class SEGYDataModule(pl.LightningDataModule) : 
   
    def __init__(self, segy_filename , config_path) : # pass config file using get_config

        super().__init__()

        # full configuration 
        self.config_path = config_path
        self.config = get_config(config_path)

        self.data_dir = os.path.join(self.config['dir']['workspace'],self.config['dir']['data']) 

        self.segy_filename = segy_filename 
        self.batch_size = self.config['patch_based_training']['batch_size']
        self.noise_factor = self.config['patch_based_training']['noise_factors']
        self.split_factor = self.config['patch_based_training']['split_factor']

        
        self.patch_size = self.config['patch_based_training']['patch_size'] 
        
        self.dims = (self.batch_size,1, self.patch_size, self.patch_size)
        
        # init 
        self.train = None
        self.test = None 
        self.val = None

    def prepare_data(self) : # do not assign state here 

        # convert segy file to seisnc xarray, extract global statistics, create patch index

        prepare_data_xarray(self.config_path, self.segy_filename)
        calculate_global_stats(self.segy_filename, self.config_path)
        create_patch_index(self.segy_filename, self.config_path)

        # gc.collect()
        return
           
    def setup(self, stage=None) : 

        # print("DataModule setup phase")

        if stage == 'fit' or stage == None: 

            dataset_train = SEGYDataset(segy_filename = self.segy_filename, config_path = self.config_path, mode='train', transform=ToTensor())
            dataset_val = SEGYDataset(segy_filename = self.segy_filename, config_path = self.config_path, mode='val', transform=ToTensor())
            # print('setup fit phase completed')
            self.train , self.val = dataset_train, dataset_val
           
        if stage == 'test' : 
            dataset_test = SEGYDataset(segy_filename = self.segy_filename, config_path = self.config_path, mode='test', transform=ToTensor())
            self.test = dataset_test
            
        if stage == 'predict': 
            pass

    def train_dataloader(self) : 
        num_workers = multiprocessing.cpu_count()//2
        # print('Initialising train dataloader')

        return DataLoader(self.train, self.batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn,)
    
    def val_dataloader(self) : 
        num_workers = multiprocessing.cpu_count()//2
        # print('Initialising val dataloader')

        return DataLoader(self.val, self.batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn)

    def test_dataloader(self) : 
        num_workers = multiprocessing.cpu_count()//2
        # print('Initialising test dataloader')

        return DataLoader(self.test, self.batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn)
    
    def predict_dataloader(self) : 
        # dataset = np.load(os.path.join(self.segy_folder, 'noisy.npy'))
        # dataset = np.expand_dims(dataset, axis=0)
        # dataset = np.expand_dims(dataset, axis=0)
        # return DataLoader(dataset, num_workers=multiprocessing.cpu_count()//2)
        pass
    
if __name__ == '__main__' : 
    config_path = '../config/config_dncnn.yaml'
    segy_filename = 'Netherlands_F3_raw.sgy'
 

    # dm = SEGYDataModule(segy_filename, config_path)
    # gc.collect()
    # prepare_data_xarray(config_path, segy_filename)
    # gc.collect()
    # calculate_global_stats(segy_filename, config_path)
    # gc.collect()
    create_patch_index(segy_filename, config_path)
    # gc.collect()
    dataset = SEGYDataset(segy_filename = segy_filename, config_path = config_path, mode='train')
    dataloader = DataLoader(dataset, 32, num_workers=multiprocessing.cpu_count()//2, worker_init_fn=worker_init_fn, pin_memory=True)
    
    batch = next(iter(dataloader))
    pdb.set_trace()
