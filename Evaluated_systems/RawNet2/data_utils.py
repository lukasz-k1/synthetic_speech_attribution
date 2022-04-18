import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split


def gen_train_eval(CSV_PATH, random_state=42):
    
    df = pd.read_csv(CSV_PATH)
    labels = []
    filenames = []
    for file in df.iterrows():
        labels.append(file[1][1])
        filenames.append(file[1][0])
    

    train_filenames, eval_filenames, train_labels, eval_labels = train_test_split(filenames, labels, random_state=random_state)
        
    return train_filenames, eval_filenames, train_labels, eval_labels



def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

class Dataset_SPCUP2022_train(Dataset):
	def __init__(self, list_IDs, labels, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = dict(zip(list_IDs, labels))
            self.base_dir = base_dir
            

	def __len__(self):
           return len(self.list_IDs)


	def __getitem__(self, index):
            self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir+key, sr=16000) 
            X_pad= pad(X,self.cut)
            x_inp= Tensor(X_pad)
            y = self.labels[key]
            return x_inp, y
            
            
class Dataset_SPCUP2022_eval(Dataset):
	def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            

	def __len__(self):
            return len(self.list_IDs)


	def __getitem__(self, index):
            self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir+key, sr=16000)
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp,key     
