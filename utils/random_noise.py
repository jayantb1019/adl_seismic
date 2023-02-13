
from skimage.util import random_noise


def add_noise(data, mode='gaussian', mean=0,noise_factor=0.5, seed=42) : 
    
    if mode == 'gaussian' :  
        args = dict(mode = mode, seed=seed, clip=True, mean =mean , var = noise_factor)
        
    if mode == 'poisson' :
        args = dict(mode = mode, seed=seed, clip=True)
        
    return random_noise(data, **args)

