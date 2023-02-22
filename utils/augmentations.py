import albumentations as A 
import numpy as np 
import matplotlib.pyplot as plt 
from copy import deepcopy 

import sys 
sys.path.append('../utils/')
from random_noise import add_noise
from scipy.ndimage import rotate

'''
| Augmentation | Applicable To |
|---|---|
| Polarity Reversal | Prestack , Post stack | 
| Random Trace Dropout | Prestack , Post stack | 
| Random High Noise traces | Prestack , Post stack |
| Random Amplitude Attenuation | Pre stack, Post stack |
| Trace Shuffle | Prestack, Post stack |
| Random Amplitude Shift | Prestack , Post stack | 
|Rotate | Post stack |

An augmentation applied to clean image should also be applied to noisy image. 
Since, some of the augmentations are random in nature, the noisy image augmentation is different from clean. Hence, instead of applying same augmentation once to each, we could pass clean, noisy as a tuple and return clean_aug, noisy_aug as a tuple. 

With respect to Albumentations , both are passed into 'image' type

Ref : /notebooks/faciesmark_data_augmentation.ipynb

'''

# Polarity Reversal

def polarity_reversal(clean_noisy_tuple ,**kwargs) : 
    
    return -1 * clean_noisy_tuple[0] , -1 * clean_noisy_tuple[1]


# Random Trace Dropout 

def random_trace_dropout(patch, **kwargs) : # drops one trace
    clean, noisy = patch
    
    pct_dropouts = 0.05
    
    for i in range(int(pct_dropouts * patch.shape[1])) : 
        trace_no = np.random.randint(clean.shape[1])

        clean[: , trace_no] = 0 
        noisy[: , trace_no] = 0 
    
    return clean, noisy

# Random high noise trace 

def random_high_noise_trace(patch, **kwargs) : # adds random noise to a few traces
    clean, noisy = patch
    pct_high_noise = 0.05
    
    for i in range(int(pct_high_noise * clean.shape[1])) : 
        trace_no = np.random.randint(patch.shape[1])
        trace_data_clean = clean[:, trace_no]
        trace_data_noisy = noisy[:, trace_no]
        clean[:, trace_no] = add_noise(trace_data_clean, noise_factor=0.2)
        noisy[:, trace_no] = add_noise(trace_data_noisy, noise_factor=0.2)
    
    return clean, noisy


# Random amplitude attenuation 

def random_amp_attenuation(patch, **kwargs) : # adds random noise to a few traces
    clean, noisy = patch
    pct_atten = 0.1 # 10 % of points
    amp_scaler = 0.5 # (0, 0.9)
    
    random_inlines = np.random.randint(0,clean.shape[1], int(pct_atten * clean.shape[1]))
    random_twt = np.random.randint(0,clean.shape[0] , int(pct_atten * clean.shape[1]) )
    
    # print(random_inlines, random_twt)
    for twt,iline in zip(random_twt, random_inlines) : 
        clean[twt,iline] =  clean[twt,iline] / amp_scaler
        noisy[twt,iline] =  noisy[twt,iline] / amp_scaler
    
    return clean, noisy

# Random Trace Shuffle

def random_trace_shuffle(patch, **kwargs) : # adds random noise to a few traces
    
    clean, noisy = patch
    pct_atten = 0.05 # 10 % of tracces
    neighborhood = 4
    
    shuffle_mask = np.zeros_like(clean)
    
    def _find_iline(clean, iline_orig, neighborhood) : 
        iline_new = None
        while 1 : 
            iline_new = np.random.randint(-neighborhood,  neighborhood)
            iline_new = iline_orig + iline_new 
            
            if iline_new >= clean.shape[1] : 
                continue 
            else : 
                break 
        return iline_new
    
    random_inlines = np.random.randint(0,patch.shape[1], int(pct_atten * patch.shape[1]))
    
    for iline in random_inlines : 
        shuffle_mask[:, iline] = -1
        
        iline_new = _find_iline(clean, iline, neighborhood)
        
        shuffle_mask[:, iline_new] = 1
        
        new_data = clean[:,iline_new]
        clean[:,iline_new] =  clean[:,iline]
        clean[:,iline] = new_data

        new_data = noisy[:,iline_new]
        noisy[:,iline_new] =  noisy[:,iline]
        noisy[:,iline] = new_data
    
    
    return clean, noisy # return shuffle mask if needed

# Horizontal Flip 

def horizontal_flip(patch ,**kwargs) : 
    clean, noisy = patch
    return np.flip(clean, axis=-1), np.flip(noisy, axis=-1)

# Random Amplitude Shift 

def random_amp_shift(patch, **kwargs) : # adds random noise to a few traces
    clean, noisy = patch
    pct_atten = 0.5 # 10 % of points
    
    mask = np.zeros_like(patch)
    
    random_inlines = np.random.randint(0,patch.shape[1], int(pct_atten * patch.shape[1]))
    random_twt = np.random.randint(0,patch.shape[0] , int(pct_atten * patch.shape[1]) )
    
    # print(random_inlines, random_twt)
    for twt,iline in zip(random_twt, random_inlines) : 
        
        mask[twt,iline] = 1
        
        rand_amp_shift = np.random.choice([-1,1]) * 0.1 * np.random.randint(0,9)
        
        clean[twt,iline] =  np.clip(clean[twt,iline] + rand_amp_shift , -0.9, 0.9 )
        noisy[twt,iline] =  np.clip(noisy[twt,iline] + rand_amp_shift , -0.9, 0.9 )
    
    return clean, noisy

# Vertical Flip omitted 



# Rotation 

def rotate_patch(patch, **kwargs) : 
    clean, noisy = patch
    rot_angle = np.random.choice([-5,-2.5, 2.5,5])

    rotated_clean = rotate(clean, angle = rot_angle , reshape=False )
    rotated_noisy = rotate(noisy, angle = rot_angle , reshape=False )
    # return np.clip(clipped_zoom(rotated, 1.2),-1,1)
    return np.clip(rotated_clean, -1,1) , np.clip(rotated_noisy, -1,1)




# Tests 

## Helper functions 
def patch_stats(patch) : 
        return f'mean : {np.mean(patch)} , std : {np.std(patch)} , min : {np.min(patch)} , max : {np.max(patch)}'
    
def visualize(patch, patch_aug) : # patch = [clean, noisy] , patch_aug = [clean_aug, noisy_aug]
    kwargs = dict(cmap='seismic', vmin = -1, vmax = 1, aspect='auto')
    fig, ax = plt.subplots(ncols=4, figsize=(4 * 4,4))
    ax[0].imshow(patch[0], **kwargs)
    ax[0].set_title('Clean Original')
    ax[1].imshow(patch_aug[0], **kwargs)
    ax[1].set_title('Clean Augmented')
    ax[2].imshow(patch[1], **kwargs)
    ax[2].set_title('Noisy Original')
    ax[3].imshow(patch_aug[1], **kwargs)
    ax[3].set_title('Noisy Augmented')
    print('clean stats'  , patch_stats(patch[0]), patch_stats(patch[1]))
    print('noisy stats'  , patch_stats(patch_aug[0]), patch_stats(patch_aug[1]))
    plt.show()

if __name__ == '__main__' : 
    # dataset path 
    dataset_path = '../data/faciesmark/raw/seismic_entire_volume.npy'
    patchsize = 128
    stride = 128
    info_path = '../data/faciesmark/train_patches_32_32_2d.csv'
    mode = '2d'

    dataset = np.load(dataset_path)

    random_xline_start  = np.random.randint(0, 901 - patchsize)
    random_twt_start = np.random.randint(0,255 - patchsize)
    patch_clean = dataset[0, random_xline_start : random_xline_start + patchsize , random_twt_start : random_twt_start + patchsize].T
    patch_clean.shape

    patch_noisy  = add_noise(patch_clean, noise_factor=0.01)
    
    
        
    transforms = A.Compose([
    A.Lambda(name='polarity_reversal', image=polarity_reversal, p=1)])

    aug1 = transforms(image = np.stack([patch_clean, patch_noisy], axis=0))

    visualize([patch_clean, patch_noisy], aug1['image'])