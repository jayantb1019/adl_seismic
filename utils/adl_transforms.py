
import torch
from torchvision import transforms
import numpy as np

import albumentations as A 

# class ToTensor_fn(object) : 
#     def __call__(self, sample) : 
#         for k in sample.keys():
#             if (sample[k] is not False) and (not isinstance(sample[k], str)):
#                 sample[k] = torch.from_numpy(sample[k])
#         return sample

class AbsMaxScaling(object) : 
    def __init__(self, abs_max) :
        super().__init__()
        self.abs_max = abs_max
        
    def __call__(self, sample) : 
        for k in sample.keys():
            if (sample[k] is not False) and (not isinstance(sample[k], str)):
                sample[k] = torch.clip(sample[k], - self.abs_max, self.abs_max) / self.abs_max
        return sample
    
class PolarityReversal(object) : 
    pass

class RandomTraceMute(object) : 
    pass
    
    
def training_transforms() : 

    '''
    Applicable transforms : 
    AbsMaxScaling
    Polarity Reversal 
    Random Trace Mute
    Random High Noise traces
    Random Amplitude Attenuation
    Trace Shuffle 
    Random Amplitude Shift --> shift by dc and clip
    Random Low Pass Filter 
    Random High Pass Filter
    Zoom Rotate 
    
    ToTensor

    '''
    pass 

def test_transforms() : 
    '''
    Applicable Transfroms
    
    AbsMaxScaling
    To Tensor
    '''
    pass
    
    

