
from skimage.util import random_noise
import torch
from torch.distributions import Uniform, Poisson


def add_noise(data, mode='gaussian', mean=0,noise_factor=0.5, seed=42) : 
    
    if mode == 'gaussian' :  
        args = dict(mode = mode, seed=seed, clip=True, mean =mean , var = noise_factor)
        
    if mode == 'poisson' :
        args = dict(mode = mode, seed=seed, clip=True)
        
    return random_noise(data, **args)


def add_poisson(
    tensor, 
    lam,
    inplace: bool = False,
    clip: bool = True,
) :
    """Adds Poisson noise to a batch of input images.
    Args:
        tensor (Tensor): Tensor to add noise to; this should be in a B*** format, e.g. BCHW.
        lam (Union[Number, Tuple[Number, Number]]): Distribution rate parameter (lambda) for
            noise being added. If a Tuple is provided then the lambda is pulled from the
            uniform distribution between the two value is used for each batched input (B***).
        inplace (bool, optional): Whether to add the noise in-place. Defaults to False.
        clip (bool, optional): Whether to clip between image bounds (0.0-1.0 or 0-255).
            Defaults to True.
    Returns:
        Tuple[Tensor, Union[Number, Tensor]]: Tuple containing:
            * Copy of or reference to input tensor with noise added.
            * Lambda used for noise generation. This will be an array of the different
            lambda used if a range of lambda are being used.
            
    Ref : https://github.com/COMP6248-Reproducability-Challenge/selfsupervised-denoising/blob/master-with-report/ssdn/ssdn/utils/noise.py
    """
    if not inplace:
        tensor = tensor.clone()

    if isinstance(lam, (list, tuple)):
        if len(lam) == 1:
            lam = lam[0]
        else:
            assert len(lam) == 2
            (min_lam, max_lam) = lam
            uniform_generator = Uniform(min_lam, max_lam)
            shape = [tensor.shape[0]] + [1] * (len(tensor.shape) - 1)
            lam = uniform_generator.sample(shape)
    tensor.mul_(lam)
    poisson_generator = Poisson(torch.tensor(1, dtype=float))
    noise = poisson_generator.sample(tensor.shape)
    tensor.add_(noise)
    tensor.div_(lam)
    if clip:
        tensor = tensor.clamp(-1,1)

    return tensor, lam