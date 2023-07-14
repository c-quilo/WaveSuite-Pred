#!/usr/bin/env python
# coding: utf-8


#class build_WaveSuite was taken and modified from here: 
# https://github.com/deepfindr/gvae/blob/master/dataset.py


import numpy as np
import torch 
import vtktools 
import pandas as pd  

import os   ## import os.path as osp
import re
from tqdm import tqdm

from torch_geometric.data import Data, Dataset

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

import random 

# In[ ]: Scaler functions 


def scaler(x, xmin, xmax, minimum, maximum):
    scale = (maximum - minimum) / (xmax - xmin)
    xScaled = scale * x + minimum - xmin * scale
    return xScaled

def inverseScaler(xscaled, xmin, xmax, minimum, maximum):
    scale = (maximum - minimum) / (xmax - xmin)
    xInv = (xscaled/scale) - (minimum/scale) + xmin
    return xInv


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = False

    return True


# In[ ]:


def save_raw_node_features(filename, features_list, original_folder, raw_folder):
    
    file_location = original_folder + '/' + filename # Apparently there is no difference between '/' and '\'
    vtu_object = vtktools.vtu(file_location)
    
    node_features = vtu_object.GetField(features_list[0]).T

    for i in range(1,len(features_list)):
        next_feature = vtu_object.GetField(features_list[i]).T
        node_features = np.append(node_features,next_feature,axis=0)

    node_features = node_features.T
    #node_features = torch.tensor(node_features)
    
    save_as = raw_folder + '/' + filename[:-4] + '.npy' # Add '.pt' instead of '.npy' for torch
    np.save(save_as,node_features)
    #torch.save(node_features, save_as)
    
    
    
# In[ ]: Process set


def process_set(raw_paths_list , test, 
                xmin_orig, xmax_orig, minimum_scaler, maximum_scaler,
                fixed_edge_index, 
                pre_filter, pre_transform, 
               processed_dir_location):
    
    #(raw_paths_list=self.raw_paths , test=self.test, 
    #            xmin_orig=self.xmin_orig, xmax_orig=self.xmax_orig, minimum_scaler=self.minimum_scaler, maximum_scaler=self.maximum_scaler,
    #            fixed_edge_index=self.fixed_edge_index, 
    #            pre_filter=self.pre_filter, pre_transform=self.pre_transform, 
    #           processed_dir_location=self.processed_dir):
        
        for jj, raw_path in enumerate(raw_paths_list):
            
            node_features = np.load(raw_path)
            node_features = scaler(node_features, xmin_orig, xmax_orig, minimum_scaler, maximum_scaler)
            node_features = torch.tensor(node_features) 
            
            # Read data from `raw_path`
            data = Data(x=node_features, edge_index=fixed_edge_index) 

            if pre_filter is not None and not pre_filter(data):
                continue

            if pre_transform is not None:
                data = pre_transform(data)
            
            wave_filename = f'data_{test}_{jj}.pt'
            torch.save(data, os.path.join(processed_dir_location, wave_filename )) #f'data_{self.length}.pt'
            

    
    
# In[ ]: Class build_WaveSuite

 
class build_WaveSuite(Dataset):
    def __init__(self, root, original_files_list, features_list, fixed_edge_index, 
                 xmin_orig = None, xmax_orig = None, minimum_scaler=0, maximum_scaler=1,
                 # file_beginning = 'regularWave_', first_idx = 40,
                 test='train', transform=None, pre_transform=None, length=0, seed=42):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        
        set_seed(seed)
        
        self.test = test
        self.original_files_list = original_files_list
        self.length = length
        
        self.features_list = features_list
        #self.file_beginning = file_beginning
        #self.first_idx = first_idx
        
        self.fixed_edge_index = fixed_edge_index
        self.xmin_orig = xmin_orig
        self.xmax_orig = xmax_orig
        self.minimum_scaler = minimum_scaler
        self.maximum_scaler = maximum_scaler
        
        super(build_WaveSuite, self).__init__(root, transform, pre_transform)
        
        print('self.indices() : ', self.indices() )
        
        #This Dataset package is so awesome, that this variables are created in the super class without any extra coding
        #print('self.raw_dir: ', self.raw_dir, '\n')
        #print('self.processed_dir: ', self.processed_dir, '\n')
        #print('self.raw_paths: ', self.raw_paths, '\n')
        #print('self.processed_paths: ', self.processed_paths, '\n')
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        raw_files_list = [ (file[:-4]+ '.npy') for file in self.original_files_list]
        #print('raw_files_list: ', raw_files_list, '\n')
        return raw_files_list

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped """
        #processed_files_list = [ (file[:-4]+ '.pt') for file in self.original_files_list]
        # print('processed_files_list: ', processed_files_list, '\n')
        
        print('RETURN list:  processed_file_names')
        
        set_seed(42)
        
        processed_files = [f for f in os.listdir(self.processed_dir) if not f.startswith("pre")]
        
        if self.test=='train':
            processed_files = [file for file in processed_files if "train" in file]
            if len(processed_files) == 0:
                return ["no_files.dummy"]
            
            #last_file = sorted(processed_files)[-1]
            #index = int(re.search(r'\d+', last_file).group()) 
            index = len(processed_files) 
            self.length = index
            return [f'data_train_{i}.pt' for i in list(range(0, index))]
        
        elif self.test=='validation':
            processed_files = [file for file in processed_files if "validation" in file]
            if len(processed_files) == 0:
                return ["no_files.dummy"]
            
            #last_file = sorted(processed_files)[-1]
            #index = int(re.search(r'\d+', last_file).group())
            index = len(processed_files) 
            self.length = index
            return [f'data_validation_{i}.pt' for i in list(range(0, index))]
        
        elif self.test=='test':
            processed_files = [file for file in processed_files if "test" in file]
            if len(processed_files) == 0:
                return ["no_files.dummy"]
            
            #last_file = sorted(processed_files)[-1]
            #index = int(re.search(r'\d+', last_file).group())
            index = len(processed_files)  
            self.length = index
            return [f'data_test_{i}.pt' for i in list(range(0, index))]
        
        else:
            print('ERROR. Please enter one of the following options for test: [train, validation, test]')
            # processed_files_list = [ f'data_{idx}.pt' for idx in range( len(self.original_files_list) ) ]
            if len(processed_files) == 0:
                return ["no_files.dummy"]
            self.length = len(processed_files)
            return processed_files      
        

    def download(self):
        
        print('Download started \n ')
        
        orig_dir = self.raw_dir# + '\..'
        
        original_folder = orig_dir
        raw_folder = self.raw_dir
        
        for file in self.original_files_list:
            save_raw_node_features(file, self.features_list, original_folder, raw_folder)
            
        print('Download finished \n ')

    def process(self):
        
        print('Processing started \n ')
        
        raw_paths_copy1 = self.raw_paths.copy()
        num_raw_paths = len(raw_paths_copy1)

        permutation0 = np.random.permutation(num_raw_paths)
        raw_paths_copy2 = [raw_paths_copy1[entry] for entry in permutation0]

        raw_paths_train = raw_paths_copy2[:int(num_raw_paths * 0.8)]
        raw_paths_validation = raw_paths_copy2[int(num_raw_paths * 0.8):int(num_raw_paths * 0.9)]
        raw_paths_test = raw_paths_copy2[int(num_raw_paths * 0.9):]
        
        #Create all files: train, validation and test 
        process_set(raw_paths_list=raw_paths_train , test='train', 
                    xmin_orig=self.xmin_orig, xmax_orig=self.xmax_orig, 
                    minimum_scaler=self.minimum_scaler, maximum_scaler=self.maximum_scaler,
                    fixed_edge_index=self.fixed_edge_index, pre_filter=self.pre_filter, pre_transform=self.pre_transform, 
                    processed_dir_location=self.processed_dir)
        
        process_set(raw_paths_list=raw_paths_validation , test='validation', 
                    xmin_orig=self.xmin_orig, xmax_orig=self.xmax_orig, 
                    minimum_scaler=self.minimum_scaler, maximum_scaler=self.maximum_scaler,
                    fixed_edge_index=self.fixed_edge_index, pre_filter=self.pre_filter, pre_transform=self.pre_transform, 
                    processed_dir_location=self.processed_dir)
        
        process_set(raw_paths_list=raw_paths_test , test='test', 
                    xmin_orig=self.xmin_orig, xmax_orig=self.xmax_orig, 
                    minimum_scaler=self.minimum_scaler, maximum_scaler=self.maximum_scaler,
                    fixed_edge_index=self.fixed_edge_index, pre_filter=self.pre_filter, pre_transform=self.pre_transform, 
                    processed_dir_location=self.processed_dir)
        
        #Save CSV indicating which files were used in each dataset
        just_train = ['train' for i in raw_paths_train]
        just_validation = ['validation' for i in raw_paths_validation]
        just_test = ['test' for i in raw_paths_test]
        
        just_names = just_train+just_validation+just_test
        
        data = {'col_1': just_names , 'col_2': (raw_paths_train+raw_paths_validation+raw_paths_test) }
        save_paths = pd.DataFrame.from_dict(data)
        save_paths.to_csv('saved_paths.csv',index=False)
        
        # self.length += 1
        if self.test=='train':
            self.length = len(raw_paths_train)
        elif self.test=='validation':
            self.length = len(raw_paths_validation)
        elif self.test=='test':
            self.length = len(raw_paths_test)
        else:   
            print('ERROR 2. Please enter one of the following options for test: [train, validation, test]')

        print(f"Done. Stored {self.length} wavesuite files.")

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.length

    def get(self, idx):
        """ 
        - Equivalent to __getitem__ in pytorch
        - Is not needed for PyG's InMemoryDataset
        """
        
        if self.test=='train':
            wave_filename = f'data_train_{idx}.pt'
        elif self.test=='validation':
            wave_filename = f'data_validation_{idx}.pt'
        elif self.test=='test':
            wave_filename = f'data_test_{idx}.pt'
        else:   
            print('ERROR 2. Please enter one of the following options for test: [train, validation, test]')
        
        data = torch.load(os.path.join(self.processed_dir, wave_filename ))        
        return data