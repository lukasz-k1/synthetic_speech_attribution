from os.path import join, getsize
from os import walk
from tqdm import tqdm
from scipy import signal
import numpy as np
import random
from cqcc import cqcc
from lfcc import lfcc
from scipy.signal import stft, istft
import librosa
import pandas as pd


def get_windowed_signal(data, fs, window_type, window_length, seed):
    """Return randomly chosen, windowed signal slice using scipy window types

    Args:
        data (array): input signal to be windowed
        fs (int): sample rate of the signal
        window_type (string): scipy defined window type passed to get_window
        window_length (float): length of window in seconds
        seed (int): random seed for reproducibility

    Returns:
        array: array of windowed signal samples
    """
    random.seed(seed)

    data = np.array(data)
    sc = len(data)
    window_sample_count = window_length*fs

    assert(sc >= window_sample_count), "Window cannot be longer than signal!"

    if window_sample_count != sc:
        starting_sample = random.choice(range(0, sc-window_sample_count))
    else:
        starting_sample = 0

    win = signal.get_window(window_type, window_sample_count)
    data = np.multiply(
        data[starting_sample:starting_sample+window_sample_count], win)
    return data


def find_longest_file(DATA_PATH):
  '''
    Returns the longest file and filename from specified path

      Parameters:
        DATA_PATH (str): Path where data is stored

      Returns:
        longest_file (numpy.array): Audio data of the longest file 
        longest_file_path (str): Filename of the longest file
  '''
  wav_paths = []
  for directory, subdirs, files in walk(DATA_PATH):
    wav_paths.extend([join(directory, file) for file in files if file.endswith('.wav')])
  longest_file_path = max(wav_paths, key=lambda x: getsize(x))
  longest_file, fs = librosa.load(longest_file_path, sr=None)
  return longest_file, longest_file_path

def get_csv_labels(CSV_PATH):
  '''
    Returns labels which are stored in CSV spreadsheet

      Parameters:
        CSV_PATH (str): Path where spreadsheet with labels is stored

      Returns:
        labels (list): Labels from CSV
  '''
  df = pd.read_csv(CSV_PATH)
  labels = []
  for file in df.iterrows():
      labels.append(file[1][1])
  return labels

def spectrogram_inverse(y, sr):
  '''
    Returns audio file with inverted frequency response (spectrogram flipped vertically)

      Parameters:
        y (numpy.array): Audio file whose frequency response is to be inverted
        y (int): Sample rate

      Returns:
        y_istft (numpy.array): Audio file with inverted frequency response
  '''
  y = y[0:int(np.floor(len(y)/128)*128)]

  f, t, y_stft = stft(y, sr, nperseg=256, noverlap=128, nfft=512, return_onesided=True)

  y_stft = np.flip(y_stft, 0)

  t_, y_istft = istft(y_stft, sr, nperseg=256, noverlap=128, nfft=512)

  return y_istft

def features_extraction(CSV_PATH=None, DATA_PATH=None, features_type="MFCC", delta=0, windowing=True, longest_file_size=None, **feature_params):
  '''
    Returns extracted features from specified audio files stored in DATA_PATH, labels must be in CSV file stored in CSV_PATH

      Parameters:
        DATA_PATH (str): Path where data is stored
        CSV_PATH (str): Path where spreadsheet with labels is stored
        features_type (str): {'MFCC', LFCC', CQCC', IMFCC', ILCC', ICQCC'} Type of feature to extract
        delta (int): Number of deltas to add at the end of feature
        windowing (bool): If True file is randomly cut to the length of 1s. If False all files are extended (by mirroring) to the length of the longest file in dataset
        longest_file_size (int): Number of samples in the longest file (if windowing=False it is not needed)
        **feature_params : keyworded arguments for specified feature

      Returns:
        X (numpy.ndarray): Array with computed features for all files
  '''
  df = pd.read_csv(CSV_PATH)
  features = []
  for file in tqdm(df.iterrows(), total=df.shape[0]):
      x, fs = librosa.load(join(DATA_PATH, file[1][0]), sr=None)
      
      if(windowing):
          x=get_windowed_signal(x, fs, "hamm", 1, 42)
      else:
          x = librosa.util.fix_length(x, size=longest_file_size, mode='symmetric')
      
      feature = None
      if features_type=="MFCC":
        feature = librosa.feature.mfcc(y=x, sr=fs, **feature_params)[1:]

      elif features_type=="CQCC":
        feature = cqcc(signal=x, sr=fs, **feature_params)[2][1:]
      
      elif features_type=="LFCC":
        feature = lfcc(y=x, sr=fs, **feature_params)[1:]
      
      elif features_type=="IMFCC":
        x = spectrogram_inverse(x, fs)
        feature = librosa.feature.mfcc(y=x, sr=fs, **feature_params)[1:]

      elif features_type=="ICQCC":
        x = spectrogram_inverse(x, fs)
        feature = cqcc(signal=x, sr=fs, **feature_params)[2][1:]
      
      elif features_type=="ILFCC":
        x = spectrogram_inverse(x, fs)
        feature = lfcc(y=x, sr=fs, **feature_params)[1:]
      
      delta_tmp = feature

      if delta>0:
        for i in range(delta):
          delta_tmp = librosa.feature.delta(delta_tmp, width=5, order=1, mode='mirror')
          feature = np.concatenate((feature, delta_tmp), axis=1)
   
      features.append(feature)

  X = np.stack([feature for feature in features], axis=0)
  X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
  
  return X

def coeffs_create(CSV_PATH, DATA_PATH, best_MFCC_params, best_LFCC_params, best_CQCC_params):
  '''
    Saves numpy arrays with extracted features (in current path with extension .npy) from 
    specified audio files stored in DATA_PATH, labels must be in CSV file stored in CSV_PATH

      Parameters:
        DATA_PATH (str): Path where data is stored
        CSV_PATH (str): Path where spreadsheet with labels is stored
        best_MFCC_params (dict): keyworded arguments for MFCC
        best_LFCC_params (dict): keyworded arguments for LFCC
        best_CQCC_params (dict): keyworded arguments for CQCC
      
      Returns:
        None
  '''
  print("MFCC")
  X_MFCC = features_extraction(CSV_PATH=CSV_PATH, features_type="MFCC", DATA_PATH=DATA_PATH, windowing=True, **best_MFCC_params)
  np.save('X_MFCC.npy', X_MFCC)
  del X_MFCC

  print("LFCC")
  X_LFCC = features_extraction(CSV_PATH=CSV_PATH, features_type="LFCC", DATA_PATH=DATA_PATH, windowing=True, **best_LFCC_params)
  np.save('X_LFCC.npy', X_LFCC)
  del X_LFCC

  print("CQCC")
  X_CQCC = features_extraction(CSV_PATH=CSV_PATH, features_type="CQCC", DATA_PATH=DATA_PATH, windowing=True, **best_CQCC_params)
  np.save('X_CQCC.npy', X_CQCC)
  del X_CQCC

  print("IMFCC")
  X_IMFCC = features_extraction(CSV_PATH=CSV_PATH, features_type="IMFCC", DATA_PATH=DATA_PATH, windowing=True, **best_MFCC_params)
  np.save('X_IMFCC.npy', X_IMFCC)
  del X_IMFCC

  print("ILFCC")
  X_ILFCC = features_extraction(CSV_PATH=CSV_PATH, features_type="ILFCC", DATA_PATH=DATA_PATH, windowing=True, **best_LFCC_params)
  np.save('X_ILFCC.npy', X_ILFCC)
  del X_ILFCC

  print("ICQCC")
  X_ICQCC = features_extraction(CSV_PATH=CSV_PATH, features_type="ICQCC", DATA_PATH=DATA_PATH, windowing=True, **best_CQCC_params)
  np.save('X_ICQCC.npy', X_ICQCC)
  del X_ICQCC

def coeffs_load(coeff_type=None):
  '''
    Returns specified coefficients which are stored in current directory

      Parameters:
        coeff_type (str): {'MFCC', LFCC', CQCC', IMFCC', ILCC', ICQCC'} Type of feature to load

      Returns:
        (numpy.ndarray): Array with specified features
  '''
  if coeff_type=="MFCC":
    return np.load('X_MFCC.npy')
  elif coeff_type=="LFCC":
    return np.load('X_LFCC.npy')
  elif coeff_type=="CQCC":
    return np.load('X_CQCC.npy')
  elif coeff_type=="IMFCC":
    return np.load('X_IMFCC.npy')
  elif coeff_type=="ILFCC":
    return np.load('X_ILFCC.npy')
  elif coeff_type=="ICQCC":
    return np.load('X_ICQCC.npy')