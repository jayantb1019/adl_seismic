import os 
import shutil
import yaml
import gc 

import pdb

import numpy as np
import pandas as pd

import h5py

from segysak.segy import segy_header_scan, segy_converter
from segysak import open_seisnc
from dask.distributed import Client



# default segy byte location configuration 


def generate_xarray_from_segy(segy_path, seisnc_path, byte_locations) : 
    # generate a seisnc
    print('Generating seisnc xarray at Path :', seisnc_path)
    segy_converter(segy_path, seisnc_path, iline = byte_locations['INLINE_3D'], xline=byte_locations['CROSSLINE_3D'], cdpx=byte_locations['CDP_X'], cdpy = byte_locations['CDP_Y'])

def derive_byte_locations(segy_path, defaults) : 
    
    # default byte locations are loaded from config, compared with scanned headers

    print('Scanning Headers for byte locations of ILINE,XLINE,CDPX, CDPY')

    # scan headers 
    scanned_headers = segy_header_scan(segy_path)
    scanned_byte_locations = scanned_headers.to_dict()
    
    params = defaults.keys()

    for key in params : # if scanned headers contain a byte location, replace the defaults with scanned ones.
        if key in scanned_byte_locations.keys() : 
            defaults[key] = scanned_byte_locations[key]
    gc.collect()
    return defaults
            
def load_seisnc(seisnc_path,chunk_direction = 'iline', chuncksize = 100) : 
    # print("Loading seisnc xarray")
    chunks = dict(chunk_direction = chuncksize)
    seisnc = open_seisnc(seisnc_path, chunks = chunks)

    gc.collect()

    return seisnc

def calculate_global_stats(segy_filename, config_path) :

    segy_folderpath = derive_segy_folderpath(segy_filename,config_path)
    
    # read config file 
    config = get_config(config_path)

    # derive path to store global statistics 
    global_stats_filename = config['dir']['global_stats']
    global_stats_path = os.path.join(segy_folderpath, global_stats_filename)

    seisnc_path = derive_seisnc_path(segy_filename, config_path)
    
    

    if not os.path.exists(global_stats_path) : 
        # calculate global statistics

        print('Calculating global statistics')

        # load seisnc 
        gc.collect()
        seisnc = load_seisnc(seisnc_path)

        # global statistics 
        quantiles = (seisnc.__abs__()).quantile(.99, dim = ['xline', 'twt'])
        
        stat_global_mean = seisnc.mean().compute().data.values
        stat_global_std = seisnc.std().compute().data.values
        stat_abs_max = max(quantiles.compute().data.values)

        # write to file
        write_global_stats(global_stats_path, stat_global_mean, stat_global_std, stat_abs_max)
    else : 
        print("Global Statistics already exists")




def write_global_stats(global_stats_path,mean, std, abs_max) : 
    print('Writing Global Statistics to file')

    with h5py.File(global_stats_path, 'w') as f : 
        dataset = f.create_dataset('stats', shape=(1,3), dtype='f')
        dataset[...] = mean,std,abs_max
    f.close()


def prepare_data_xarray(config_path, segy_filename) : 

    # pdb.set_trace()

    # read config file 
    config = get_config(config_path)
    
    workspace_dir = config['dir']['workspace']
    data_dir = config['dir']['data']
    
    # derive segy path
    segy_path = os.path.join(workspace_dir, data_dir, segy_filename)

    print('Input SEGY Path :', segy_path)
    
    # generate a folder to store all files 
    segy_foldername = segy_filename.split('.')[0].lower()
    segy_folderpath = os.path.join(workspace_dir, data_dir, segy_foldername)

    if not os.path.exists(segy_folderpath) : 
        print('Creating a directory for SEGY input')
        os.mkdir(segy_folderpath)
        print('Directory created at : ', segy_folderpath)
    else : 
        print('SEGY folder already exists')

    # copy segy to new path 
    new_segy_path = os.path.join(segy_folderpath, segy_filename)
    print(new_segy_path)
    if not os.path.exists(new_segy_path) : 
        print('Copying Input SEGY to new path')
        shutil.copyfile(segy_path, new_segy_path)
        print("Copying completed")
    else : 
        print('SEGY already copied to SEGY folder')
        
    # derive path to generate seisnc file
    seisnc_filename = segy_filename.split('.')[-2] + '.seisnc'
    seisnc_path = os.path.join(segy_folderpath, seisnc_filename)


    # start a dask cluster 
    client = Client()

    # convert to seisnc xarray if it doesnot exist
    if not os.path.exists(seisnc_path) :
        byte_locations = derive_byte_locations(new_segy_path, defaults=config['byte_locations'])
        generate_xarray_from_segy(new_segy_path, seisnc_path, byte_locations)

    else : 
        print('seisnc already exists')

    # # clip seisnc # it is more computationally efficient to clip patch wise in dataloader instead. 

    # seisnc_clipped = seisnc.clip(min = -1 * abs_max, max = abs_max)

    # # normalise seisnc
    # seisnc_norm = seisnc_clipped.map(lambda x : x / abs_max)



def derive_seisnc_path(segy_filename, config_path) : 

    segy_folderpath = derive_segy_folderpath(segy_filename,config_path)
    seisnc_filename = segy_filename.split('.')[-2] + '.seisnc'
    return  os.path.join(segy_folderpath, seisnc_filename)

def is_seisnic_created(segy_filename, config_path) :
    
    seisnc_path  = derive_seisnc_path(segy_filename, config_path)

    return os.path.exists(seisnc_path)

    
def derive_segy_folderpath(segy_filename, config_path) : 

    # read config file 
    config = get_config(config_path)
    
    workspace_dir = config['dir']['workspace']
    data_dir = config['dir']['data']

    # generate a folder to store all files 
    segy_foldername = segy_filename.split('.')[0].lower()
    return os.path.join(workspace_dir, data_dir, segy_foldername)

def derive_patch_index_filepaths(segy_filename, config_path) : 
    # derive segy folder path 
    segy_folderpath = derive_segy_folderpath(segy_filename, config_path)
    
    # read config file 
    config = get_config(config_path)

    # csv file for train and test patches metadata 
    train_patches_metadata_path = (os.path.join(segy_folderpath, config['dir']['train_patch_info'])).replace("$$stride$$", str(config['patch_based_training']['stride']))
    test_patches_metadata_path = (os.path.join(segy_folderpath, config['dir']['test_patch_info'])).replace("$$stride$$", str(config['patch_based_training']['stride']))
    val_patches_metadata_path = (os.path.join(segy_folderpath, config['dir']['val_patch_info'])).replace("$$stride$$", str(config['patch_based_training']['stride']))

    return train_patches_metadata_path, val_patches_metadata_path, test_patches_metadata_path 
    
def create_patch_index(segy_filename,config_path) : 
    '''
    Creates train and test patch indexes as text files with headers : patch_index, iline, xline_start, twt_start

    '''
    # pdb.set_trace()

    # check if seisnc is created 
    if not is_seisnic_created(segy_filename, config_path) : 
        prepare_data_xarray(config_path, segy_filename)
    

    # derive seisnc path 
    seisnc_path = derive_seisnc_path(segy_filename,config_path)


    # csv file for train and test patches metadata 
    train_patches_metadata_path,val_patches_metadata_path,test_patches_metadata_path = derive_patch_index_filepaths(segy_filename, config_path)

    # check if files are already generated and exit if they are generated 
    if os.path.exists(train_patches_metadata_path) and os.path.exists(test_patches_metadata_path) and os.path.exists(val_patches_metadata_path): 
        print('Patches information already exists')
        
    else : 
        print('Creating patches index for train , val and test data')
        # read seisnc 
        seisnc = load_seisnc(seisnc_path)

        # derive no of inlines 
        ilines = seisnc['iline'].values
        n_ilines = len(ilines)

        # derive nsamples 
        twt = seisnc['twt'].values 

        print(ilines.dtype, twt.dtype)

        
        # read config file 
        config = get_config(config_path)

        # split inlines into train , val, test
        split_factors = config['patch_based_training']['split_factor']
        stride = config['patch_based_training']['stride']
        patch_size = config['patch_based_training']['patch_size']

        n_ilines_train = int(n_ilines * split_factors[0])
        n_ilines_val = int(n_ilines * split_factors[1])

        ilines_train = ilines[:n_ilines_train]
        ilines_val = ilines[n_ilines_train : n_ilines_train + n_ilines_val ]
        ilines_test  = ilines[n_ilines_train + n_ilines_val : ]

        print('\ntrain inlines\n', ilines_train, '\nval inlines\n', ilines_val, '\ntest inlines \n', ilines_test)


        ## write train patches data
        train_patches_metadata_file = open(train_patches_metadata_path, 'a')

        # write table header
        train_patches_metadata_file.writelines(['patch_index, iline,xline_start, twt_start\n'])
        
        # write patch indexes
        patch_index = 0
        for iline in ilines_train : 
            xlines = seisnc.sel(iline = iline)['xline'].values # check xline numbers for each inline , coz they might vary

            for xline_start in xlines[::stride] : 

                for twt_start in twt[::stride] : 

                    train_patches_metadata_file.write(','.join([str(patch_index),str(iline),str(xline_start), str(twt_start)]) + '\n')
                    patch_index += 1

        train_patches_metadata_file.close()

        ## write test patches data 
        test_patches_metadata_file = open(test_patches_metadata_path, 'a')

        # write table header
        test_patches_metadata_file.writelines(['patch_index, iline,xline_start, twt_start\n'])
        
        # write patch indexes
        patch_index = 0
        for iline in ilines_test : 
            xlines = seisnc.sel(iline = iline)['xline'].values

            for xline_start in xlines[::stride] : 
                for twt_start in twt[::stride] : 
                    test_patches_metadata_file.write(','.join([str(patch_index),str(iline),str(xline_start), str(twt_start)]) + '\n')
                    patch_index += 1

        test_patches_metadata_file.close()


        ## write val patches data 
        val_patches_metadata_file = open(val_patches_metadata_path, 'a')

        # write table header
        val_patches_metadata_file.writelines(['patch_index, iline,xline_start, twt_start\n'])
        
        # write patch indexes
        patch_index = 0
        for iline in ilines_val : 
            xlines = seisnc.sel(iline = iline)['xline'].values

            for xline_start in xlines[::stride] : 
                for twt_start in twt[::stride] : 
                    val_patches_metadata_file.write(','.join([str(patch_index),str(iline),str(xline_start), str(twt_start)]) + '\n')
                    patch_index += 1

        val_patches_metadata_file.close()


def get_seisnc(segy_filename, config_path) : 
    seisnc_path = derive_seisnc_path(segy_filename, config_path)
    gc.collect()
    return load_seisnc(seisnc_path)

def get_patch_info(segy_filename, config_path) :
    
    train_patch_index_filepath,  val_patch_index_filepath , test_patch_index_filepath = derive_patch_index_filepaths(segy_filename, config_path)

    # column names : patch_index, iline, xline_start, twt_start

    dtypes = dict(patch_index = np.int64, iline=np.int32, xline_start = np.int32, twt_start=np.float64)

    train_patches =  pd.read_csv(train_patch_index_filepath, index_col='patch_index', dtype  = dtypes)
    val_patches =  pd.read_csv(val_patch_index_filepath, index_col='patch_index', dtype  = dtypes)
    test_patches =  pd.read_csv(test_patch_index_filepath, index_col='patch_index', dtype  = dtypes)

    return train_patches, val_patches, test_patches
    

def get_config(config_path) : 
    # read config file 
    with open(config_path, 'r') as f : 
        config = yaml.safe_load(f)
    f.close()
    return config

def get_global_stats(segy_filename, config_path) : 
    
    config = get_config(config_path)
    segy_folderpath = derive_segy_folderpath(segy_filename, config_path)

    global_stats_path = os.path.join(segy_folderpath, config['dir']['global_stats'])

    stats = None 
    f = h5py.File(global_stats_path, 'r')
    stats = f['stats'][...]
    f.close()

    return stats[0]
    

def derive_n_samples(segy_filename, config_path) : 
    
    train_patch_index_filepath, test_patch_index_filepath, val_patch_index_filepath = derive_patch_index_filepaths(segy_filename, config_path)
    
    n_train_samples = 0 
    n_test_samples = 0 
    n_val_samples = 0

    with open(train_patch_index_filepath,'r') as f : 
        n_train_samples = len(f.readlines()) - 1
    f.close()

    with open(val_patch_index_filepath,'r') as f : 
        n_val_samples = len(f.readlines()) - 1
    f.close()

    with open(test_patch_index_filepath,'r') as f : 
        n_test_samples = len(f.readlines()) - 1
    f.close()

    return n_train_samples, n_val_samples, n_test_samples


    

    
if __name__ == '__main__' : 

    config_path = '../config/config_random_denoising.yaml'
    segy_filename = 'MP41B_PSTM_STK_RND.sgy'

    prepare_data_xarray(config_path, segy_filename)
    calculate_global_stats(segy_filename, config_path)
    create_patch_index(segy_filename,config_path)