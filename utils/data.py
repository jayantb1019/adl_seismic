import os
import yaml
import numpy as np 
import pandas as pd

def get_config(config_path) : 
    # read config file 
    with open(config_path, 'r') as f : 
        config = yaml.safe_load(f)
    f.close()
    return config

def patch_index_generation(dataset_path, stride, info_path, mode = '2d') : 
    
    if not os.path.exists(dataset_path) :
        print(f'{info_path} provided for patch gen does not exist')
        return
    dataset = np.load(dataset_path)

    # exit if info_path already exists 
    if os.path.exists(info_path) : 
        print("Patch index already exists")
        return
    else : 
        # generate patches 
        n_ilines, n_xlines, n_twt = dataset.shape
        print(n_ilines, n_xlines, n_twt)
    
        patch_info_file = open(info_path, 'a')
        patch_info_file.writelines(['patch_index,iline,xline_start,twt_start\n'])
        
        # write patch indexes
        patch_index = 0
        if mode == '2d' : 
            for iline in range(n_ilines) : 
                for xline_start in range(0,n_xlines,stride) : 

                    for twt_start in range(0,n_twt,stride) :
                        
                        patch_info_file.write(','.join([str(patch_index),str(iline),str(xline_start), str(twt_start)]) + '\n')
                        
                        patch_index += 1

            patch_info_file.close()
        
        elif mode == '3d' : 
            for iline in range(0,n_ilines,stride) : 
                for xline_start in range(0,n_xlines,stride) : 

                    for twt_start in range(0,n_twt,stride) :
                        
                        patch_info_file.write(','.join([str(patch_index),str(iline),str(xline_start), str(twt_start)]) + '\n')
                        
                        patch_index += 1

            patch_info_file.close()
        
def select_patch(dataset, patch_info, index, patch_size, mode='2d') : 
    
    
    iline,xline_start, twt_start = patch_info.loc[patch_info['patch_index'] == index,['iline', 'xline_start','twt_start']].values[0]

    
    if mode == '2d' : 
        return dataset[iline, xline_start : xline_start + patch_size, twt_start : twt_start + patch_size]
    else : 
        return dataset[iline : iline + patch_size, xline_start : xline_start + patch_size, twt_start : twt_start + patch_size]
    