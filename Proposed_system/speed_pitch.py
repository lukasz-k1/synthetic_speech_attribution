from os.path import join
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from scipy.io import wavfile
import yaml
import os
import random


def speed(files_path, labels_path, destination_path):
    """
    This function changes speed of given audio file. 

    Input: 
    *   files_path - path to the folder with files we want to modify
    *   labels_path - path to the csv file with names of autio files 
        (name in first place in each column)
    *   destination_path - path to the folder, where we want to save changed files
    
    Output:
    None
    """
    

    # read labels csv file
    df = pd.read_csv(labels_path)
    new_data = []

    for file in tqdm(df.iterrows(), total=df.shape[0]): # (id, (track, algorithm))
        # read audio file
        x, fs = librosa.load(join(files_path, file[1][0]).replace('\\','/'), sr=None)

        # choose random factor for speed change (new_speed = beggining_speed * factor)
        # chooses factor from range <0.5; 0.75> U <1.25; 2.0>
        if np.random.randint(2) == 0:
            factor = (np.random.randint(25)+50)/100
        else:
            factor = (np.random.randint(75)+125)/100

        # this segment realises the time stretch of audio = changes its speed
        stretched = librosa.effects.time_stretch(x, factor)
        
        # save new audio
        dst_filename = '{}{}'.format(file[1][0][:-4],'_speed.wav')
        dst_path = join(destination_path,dst_filename).replace('\\','/')
        algorithm = file[1][1]
        
        new_data.append([dst_filename,algorithm])
        wavfile.write(dst_path, fs, stretched)

    speed_df = pd.DataFrame(new_data, columns=['track','algorithm']) 

    return speed_df

def pitch(files_path, labels_path, destination_path):
    """
    This function changes pitch of given audio file. 

    Input: 
    *   files_path - path to the folder with files we want to modify
    *   labels_path - path to the csv file with names of autio files 
        (name in first place in each column)
    *   destination_path - path to the folder, where we want to save changed files
    
    Output:
    None
    """


    # read labels csv file
    df = pd.read_csv(labels_path)
    new_data = []

    for file in tqdm(df.iterrows(), total=df.shape[0]): # (id, (track, algorithm))

        # read audio file
        x, fs = librosa.load(join(files_path, file[1][0]).replace('\\','/'), sr=None)

        # choose random number of steps (semitones) we want to modulate audio 
        # (new_pitch = beggining_pitch + steps)
        # chooses steps from range <-12; -5> U <5; 12> (only integers)
        if np.random.randint(2) == 0:
            steps = (np.random.randint(8)+5)
        else:
            steps = (np.random.randint(8)+5)/-1

        # this segment realises the pitch shifting
        shifted = librosa.effects.pitch_shift(x, sr=fs, n_steps=steps)
        
        # save new audio
        dst_filename = '{}{}'.format(file[1][0][:-4],'_pitch.wav')
        dst_path = join(destination_path,dst_filename).replace('\\','/')
        algorithm = file[1][1]
        
        new_data.append([dst_filename,algorithm])
        wavfile.write(dst_path, fs, shifted)

    pitch_df = pd.DataFrame(new_data, columns=['track','algorithm'])

    return pitch_df

# yaml read
files_path = ''
labels_path = ''
destination_path = ''

with open('data_settings.yml') as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
    
    files_path = os.path.normpath(settings['data_path'])
    destination_path = os.path.normpath('{}/{}'.format(settings['new_data_path'],'train_data'))
    labels_path = join(files_path, 'labels.csv').replace('\\','/')
    aug_labels_path = join(destination_path,'labels.csv').replace('\\','/')


# augumentation
speed_df = speed(files_path, labels_path, destination_path)
pitch_df = pitch(files_path, labels_path, destination_path)

# labels concatenate and save
aug_labels = pd.read_csv(aug_labels_path)
out_df = pd.concat([aug_labels,speed_df,pitch_df]) 

out_df.to_csv(aug_labels_path, index=False)