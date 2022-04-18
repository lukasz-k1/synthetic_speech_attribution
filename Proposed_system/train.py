from os.path import join
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import math
import os
import yaml

from python_speech_features.sigproc import preemphasis, framesig

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, GlobalAveragePooling2D, Dense, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

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

def extract_features(data_path, df, num_classes=6):
    X, y = [], []

    for file in tqdm(df.iterrows(), total=df.shape[0], desc="Processing data"):
        x, fs = librosa.load(join(data_path, file[1][0]), sr=None)
            
        for n in range(math.floor(len(x)/(3*fs))+1):
            x_n = x[3*n*fs:int((3*n+3)*fs)]
            
            if len(x_n)/fs < 1.5:
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
            stacked = np.stack((mgd, mel_spec), axis=2).astype('float32')
            X.append(stacked)
            y.append(file[1][1])
    
    X = np.array(X, dtype=np.float32)
    y = to_categorical(y, num_classes=num_classes)
    return X, y

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def res_net():
    inputs = Input(shape=(256, 299, 2))
    num_filters = 64

    t = Conv2D(kernel_size=7,
               strides=1,
               filters=num_filters,
               padding="same")(inputs)
    t = relu_bn(t)

    num_blocks_list = [2, 2, 2, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters, kernel_size=3)
        num_filters *= 2

    t = GlobalAveragePooling2D()(t)
    t = Dropout(0.2)(t)
    outputs = Dense(6, activation='softmax')(t)

    model = Model(inputs, outputs)
    
    optimizer = Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics='acc'
    )

    return model

def step_decay(epoch):
    initial_lrate = 1e-3
    drop = 0.5
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def spec_aug(spec, t_param, f_param):
    n_bins, n_frames = spec.shape
    t_start = np.random.randint(0, n_frames-t_param)
    f_start = np.random.randint(0, n_bins-f_param)
    spec[:,t_start:t_start+t_param] = 0
    spec[f_start:f_start+f_param,:] = 0
    return spec


with open('data_settings.yml') as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
    
    files_path = os.path.normpath(settings['data_path'])
    model_path = os.path.normpath(settings['save_model_path'])
    destination_path = os.path.normpath('{}/{}'.format(settings['new_data_path'],'train_data'))
    aug_labels_path = join(destination_path,'labels.csv').replace('\\','/')
    
DATA_PATH = files_path
CSV_PATH = join(DATA_PATH, 'labels.csv')
df = pd.read_csv(CSV_PATH)

# Specaugment
X, y = [], []
for file in tqdm(df.iterrows(), total=df.shape[0], desc="Processing data"):
    x, fs = librosa.load(join(DATA_PATH, file[1][0]), sr=None)
        
    for n in range(math.floor(len(x)/(3*fs))+1):
        x_n = x[3*n*fs:int((3*n+3)*fs)]
        
        if len(x_n)/fs < 1.5:
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
        mgd_aug = spec_aug(mgd, 30, 25)
        mel_spec_aug = spec_aug(mgd, 30, 25)
        stacked = np.stack((mgd_aug, mel_spec_aug), axis=2).astype('float32')
        X.append(stacked)
        y.append(file[1][1])

X_a = np.array(X, dtype=np.float32)
y_a = to_categorical(y, num_classes=6)

# Load the rest of augmented data
DATA_PATH = destination_path
CSV_PATH = aug_labels_path
df = pd.read_csv(CSV_PATH)

X, y = extract_features(DATA_PATH, df)

# Concatenate with specaug
X = np.append(X, X_a, axis=0)
y = np.append(y, y_a, axis=0)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = res_net()

lrate = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint(filepath=join(model_path, 'checkpoints/model.{epoch:02d}'))
tensorboard = TensorBoard(log_dir=join(model_path, 'logs'))
callbacks = [lrate, checkpoint, tensorboard]

history = model.fit(X, y,
                    batch_size=16,
                    epochs=40,
                    callbacks=callbacks)