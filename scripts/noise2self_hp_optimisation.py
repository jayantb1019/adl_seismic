# CONFIGURATION 

## NOISE CONFIGURATION
NOISE_TYPES = ['gaussian','mixed','poisson']
NOISE_LEVELS = [0.01,0.05,0.1,0.5]

## DNCNN HP OPTIMIZATION TRIAL RANGES
MASK_OPTIONS = [3,4,5,6,7,8]
LAYER_OPTIONS = [10,15,20]
WIDTH_OPTIONS = [16,32,64]


## Training parameters
lr = 0.001
patience_lr = 5 
patience_early_stopping = 10 
max_epochs = 100 



## INPUT
patch_size = 128 
batch_size = 1 
stride = 128 


'''
1. Select a patch
2. Create various noisy patches and store them in a folder
2. Run experiments for each noise level 
    1. 
'''

import numpy as np
import torch 
import sys 


sys.path.append('../')


def add_noise(patch, noise_type, noise_level) : 

    