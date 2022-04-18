from python_speech_features.sigproc import preemphasis, framesig
from os.path import join
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import math
import yaml
import os

def modgdf(x, n_fft=512, win_length=320, hop_length=160, window=np.hamming, preemph=0.97, lifter=6, alpha=0.3, gamma=0.3):
    # https://github.com/unisound-ail/phase-spectrum-ASR/blob/master/modgdf/extract_modgdf.py
    
    # complex stft
    x = preemphasis(x, preemph)
    frames = framesig(x, win_length, hop_length, window)
    complex_stft = np.fft.rfft(frames, n_fft)
    time_scaled_frames = np.arange(frames.shape[-1])*frames
    time_scaled_complex_stft = np.fft.rfft(time_scaled_frames, n_fft)
    mag_spec = np.abs(complex_stft)

    # cepstrally smoothing
    _spec = np.where(mag_spec == 0, np.finfo(float).eps, mag_spec)
    log_spec = np.log(_spec)
    ceps = np.fft.irfft(log_spec, n_fft)
    win = (np.arange(ceps.shape[-1]) < lifter).astype(float)
    win[lifter] == 0.5
    cepstrally_smoothed_mag_spec = np.absolute(np.fft.rfft(ceps*win, n_fft))

    # modified group delay
    real_spec = complex_stft.real
    imag_spec = complex_stft.imag
    real_spec_time_scaled = time_scaled_complex_stft.real
    imag_spec_time_scaled = time_scaled_complex_stft.imag
    divided = real_spec*real_spec_time_scaled + imag_spec*imag_spec_time_scaled
    tao = divided / (cepstrally_smoothed_mag_spec ** (2*gamma))
    abs_tao = np.absolute(tao)
    sign = 2 * (tao == abs_tao).astype(float) - 1
    #return dct(sign * (abs_tao**alpha), axis=1, norm='ortho')
    return sign * (abs_tao**alpha)

with open('data_settings.yml') as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
    
    model_path = os.path.normpath(settings['pretrained_model_path'])
    eval1_path = os.path.normpath(settings['eval_path_part1'])
    eval2_path = os.path.normpath(settings['eval_path_part2'])

model = tf.keras.models.load_model(model_path)

EVAL_PATH = eval1_path
EVAL_CSV_PATH = join(EVAL_PATH, 'labels_eval_part1.csv')

df = pd.read_csv(EVAL_CSV_PATH)
with open('part1_scores.csv', 'w') as f:
    for file in tqdm(df.iterrows(), total=df.shape[0]):
        x, fs = librosa.load(join(EVAL_PATH, file[1][1]), sr=None)
        preds_sum = []
        for n in range(math.floor(len(x)/(3*fs))+1):
            x_n = x[3*n*fs:int((3*n+3)*fs)]
            
            if len(x_n)/fs < 1:
                continue
            
            x_n = librosa.util.fix_length(x_n, size=3*fs, mode='symmetric')
            mgd = modgdf(x_n, alpha=0.3, gamma=0.3).T[:-1,:]
            mel_spec = librosa.feature.melspectrogram(y=x_n, 
                                                      sr=fs, 
                                                      n_fft=1024, 
                                                      hop_length=160, 
                                                      win_length=320, 
                                                      window='hamming',
                                                      n_mels=256)
            mel_spec = mel_spec[:,:mgd.shape[1]]
            stacked = np.stack((mgd, mel_spec), axis=2)
            stacked = np.expand_dims(stacked, axis=0)
            preds = model.predict(stacked)[0]
            preds_sum.append(preds)
        
        avg = np.array(list(map(sum, zip(*preds_sum))))/len(preds_sum)
        avg = np.exp(avg)/sum(np.exp(avg))
        
        f.write(file[1][1]+', ')
        f.write(str(np.argmax(avg)))
        f.write('\n')

EVAL_PATH = eval2_path
EVAL_CSV_PATH = join(EVAL_PATH, 'labels_eval_part2.csv')

df = pd.read_csv(EVAL_CSV_PATH)
with open('part2_scores.csv', 'w') as f:
    for file in tqdm(df.iterrows(), total=df.shape[0]):
        x, fs = librosa.load(join(EVAL_PATH, file[1][1]), sr=None)
        preds_sum = []
        for n in range(math.floor(len(x)/(3*fs))+1):
            x_n = x[3*n*fs:int((3*n+3)*fs)]
            
            if len(x_n)/fs < 1:
                continue
            
            x_n = librosa.util.fix_length(x_n, size=3*fs, mode='symmetric')
            mgd = modgdf(x_n, alpha=0.3, gamma=0.3).T[:-1,:]
            mel_spec = librosa.feature.melspectrogram(y=x_n, 
                                                      sr=fs, 
                                                      n_fft=1024, 
                                                      hop_length=160, 
                                                      win_length=320, 
                                                      window='hamming',
                                                      n_mels=256)
            mel_spec = mel_spec[:,:mgd.shape[1]]
            stacked = np.stack((mgd, mel_spec), axis=2)
            stacked = np.expand_dims(stacked, axis=0)
            preds = model.predict(stacked)[0]
            preds_sum.append(preds)
        
        avg = np.array(list(map(sum, zip(*preds_sum))))/len(preds_sum)
        avg = np.exp(avg)/sum(np.exp(avg))
        
        f.write(file[1][1]+', ')
        f.write(str(np.argmax(avg)))
        f.write('\n')