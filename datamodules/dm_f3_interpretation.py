# Ref : @Netherlands Dataset: A New Public Dataset for Machine Learning in Seismic Interpretation 
# EDA : Ref to LogSeq asset at ../assets/01-eda_1665816318067_0.ipynb

import os 
import json
import multiprocessing
import pdb 
import glob

import PIL

import numpy as np 
import pandas as pd
import h5py

from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl 



DATA_DIR = './data/f3_interpretation/'

inlines_dir = os.path.join(DATA_DIR, 'inlines')
xlines_dir = os.path.join(DATA_DIR, 'crosslines')
horizons_dir = os.path.join(DATA_DIR, 'horizons')
masks_dir = os.path.join(DATA_DIR, 'masks')
inline_patches_dir = os.path.join(DATA_DIR, 'tiles_inlines')
xline_patches_dir = os.path.join(DATA_DIR, 'tiles_crosslines')

inline_patch_labels_path = os.path.join(inline_patches_dir, 'labels.json')
xline_patch_labels_path = os.path.join(xline_patches_dir, 'label.json')

# global parameters
min_amplitude = -32767.0
max_amplitude = 32767.0

# cpu count 
cpu_count = multiprocessing.cpu_count()

# normalisation
def normalise(nparray) : 
    return (nparray - min_amplitude) / (max_amplitude - min_amplitude)

# prepare : convert individual xline ,line patches and their masks into a patches.h5 , full lines to lines.h5
patches_filename = 'patches.h5'
lines_filename = 'lines.h5'

def extract_line_no(path) : 
    return int(path.split('/')[-1].split('.')[0].split('_')[-1])

def derive_mask_path(mask_dir, line_type, line_no ) : 
    return os.path.join(mask_dir, '_'.join([line_type, str(line_no),'mask']) + '.png'  )

def derive_line_path(lines_dir, line_type, line_no ) : 
    return os.path.join(lines_dir, '_'.join([line_type, str(line_no)]) + '.tiff'  )

def generate_patch_path(row) : 
    line_type, filename = row['line_type'], row['filename']
    folder_path = inline_patches_dir if (line_type == 'inline') else xline_patches_dir
    return os.path.join(folder_path, filename + '.png')

def get_patch_data(patch_path) : 
    img = PIL.Image.open(patch_path)
    data = np.asarray(img)
    return data

def write_patch_data_to_h5(row) : 
    patches_path = os.path.join(DATA_DIR, patches_filename)
    f = h5py.File(patches_path, 'w')
    f['patches'][...] = get_patch_data(row['filepath'])
    f['labels'][...] = row['label']
    f.close()

def prepare_lines() : 
    inline_image_paths = sorted(glob.glob(inlines_dir + '/*'), key = lambda path : extract_line_no(path))
    xline_image_paths = sorted(glob.glob(xlines_dir + '/*'), key = lambda path : extract_line_no(path)) 
    mask_paths = sorted(glob.glob(masks_dir + '/*'))

    f = h5py.File(lines_filename, 'w')
    inlines = f.create_dataset('inlines')
    inline_masks = f.create_dataset('inline_masks')

    xlines = f.create_dataset('xlines')
    xline_masks = f.create_dataset('xline_masks')

    # write lines and masks to h5 
    for path in inline_image_paths : 
        inlines[...] = normalise(np.asarray(PIL.Image.open(path)))
        line_no = extract_line_no(path)
        mask_path = derive_mask_path(masks_dir, 'inline', line_no)
        inline_masks[...] = np.asarray(PIL.Image.open(mask_path))

    for path in xline_image_paths : 
        xlines[...] = normalise(np.asarray(PIL.Image.open(path)))
        line_no = extract_line_no(path)
        mask_path = derive_mask_path(masks_dir, 'crossline', line_no)
        xline_masks[...] = np.asarray(PIL.Image.open(mask_path))

def prepare_patches() : 
    inline_patch_labels = None 
    xline_patch_labels = None
    # write patches and masks to h5
    with open(os.path.join(inline_patches_dir, 'labels.json')) as f : 
            inline_patch_labels = pd.DataFrame.from_dict(json.load(f),orient='index' ).reset_index()
    f.close()

    with open(os.path.join(xline_patches_dir, 'labels.json')) as f : 
        xline_patch_labels = pd.DataFrame.from_dict(json.load(f), orient='index').reset_index()
    f.close()

    # spliting into columns 
    inline_patch_labels.columns =  ['filename', 'label']
    xline_patch_labels.columns =  ['filename', 'label']

    # deriving line and patch information
    inline_patch_labels['line_type']  = inline_patch_labels['filename'].str.split('.').str[0].str.split('_').str[0]
    inline_patch_labels['line_no']  = inline_patch_labels['filename'].str.split('.').str[0].str.split('_').str[1]
    inline_patch_labels['patch_index'] = inline_patch_labels['filename'].str.split('.').str[0].str.split('_').str[2]

    xline_patch_labels['line_type'] = xline_patch_labels['filename'].str.split('.').str[0].str.split('_').str[0]
    xline_patch_labels['line_no'] = xline_patch_labels['filename'].str.split('.').str[0].str.split('_').str[1]
    xline_patch_labels['patch_index'] = xline_patch_labels['filename'].str.split('.').str[0].str.split('_').str[2]

    patch_info = pd.concat([inline_patch_labels, xline_patch_labels], ignore_index=True).reindex()

    # converting columns to integer
    cols_to_convert = ['label', 'line_no', 'patch_index']
    for col in cols_to_convert : 
        patch_info[col] = patch_info[col].astype('int')

    patch_info = patch_info.sort_values(['line_type', 'line_no', 'patch_index'], ascending=[True, True, True])

    # generating file paths 
    patch_info['filepath'] = patch_info.apply(generate_patch_path, axis=1)
    
    # writing df to csv for reference 
    patch_info.to_csv(os.path.join(DATA_DIR,'patch_info.csv'))

    # creating h5 file and datasets 
    patches_path = os.path.join(DATA_DIR, patches_filename)
    f = h5py.File(patches_path, 'w')
    patches_dataset = f.create_dataset('patches')
    labels_dataset = f.create_dataset('labels')
    f.close()

    # writing to h5
    patch_info.apply(write_patch_data_to_h5, axis=1)




class F3Patches(Dataset) : 
    def __init__(self) :
        super().__init__()
        self.h5_file = os.path.join(DATA_DIR, patches_filename)

    def __len__(self) : 
        length = None
        with h5py.File(self.h5_file, 'r') as f : 
            length = len(f['patches'])
        f.close()
        return length

    def __getitem__(self, index) :
        f = h5py.File(self.h5_file, 'r') 
        data = np.expand_dims(f['patches'][index], 0), np.expand_dims(f['labels'][index], 0)
        f.close()
        return data

class F3DataModule(pl.LightningDataModule) : 
    def __init__(self, split_factor=0.7, batch_size = 64) : 
        super().__init__()
        self.split_factor = split_factor
        self.batch_size = batch_size

    def prepare(self) : 
        patches_path = os.path.join(DATA_DIR, patches_filename) 
        if not os.path.isfile(patches_path) : 
            prepare_patches()
    
    def setup(self, stage) : 
        self.dataset = F3Patches()

        if stage == 'fit' or None : 
            nsamples = len(self.dataset)
            train_nsamples = int(self.split_factor * nsamples)
            val_nsamples = int(0.5*(nsamples - train_nsamples))
            test_nsamples = nsamples - train_nsamples - val_nsamples 
            self.train, self.val, self.test = random_split(self.dataset, [train_nsamples, val_nsamples, test_nsamples])
        
        if stage == 'test' : 
            pass 

        if stage == 'predict' : 
            pass
    
    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True,batch_size= self.batch_size, num_workers= cpu_count)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=False,batch_size= self.batch_size, num_workers=cpu_count )

    def val_dataloader(self):
        return DataLoader(self.test, shuffle=False,batch_size= self.batch_size, num_workers=cpu_count )
    
    def val_dataloader(self):
        return DataLoader(self.dataset, shuffle=False,num_workers=cpu_count )