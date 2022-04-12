import librosa
import numpy as np
from scipy.interpolate import interpn
from scipy.fft import dct

def cqcc(signal, sr, fmin = 20, n_bins = 96, BPO = 12, n_coef = 19, TBL = 512, n_deltas = 2): 
  """
  Calculates constant Q cepstral coefficients of a signal with given sample rate

  @param signal: the input signal
  @param sr: sample rate of the input signal
  @param fmin: lower bound of analyzed frequencies
  @param n_bins: total number of frequency bins to put the spectrum into
  @param BPO: number of frequency bins per octave
  @param n_coef: number of coefficients in the final result
  @param TBL: time bin length, size of time segments in samples
  @param n_deltas: number of deltas of the signal to be included in the analysis

  @return X: two dimentional numpy array containing the CQCC
  """
  X = librosa.cqt(signal, sr, hop_length=TBL, fmin=fmin, n_bins=n_bins, bins_per_octave=BPO) 
  X = np.log10(np.abs(X)**2+0.00001)

  time_vec = np.arange(0, X.shape[1])
  time_vec = time_vec/sr*TBL
  freq_vec = np.arange(0, n_bins)
  freq_vec = fmin * (2**(freq_vec/BPO))

  target_sr = freq_vec[n_bins//2]

  ures_freq_vec = np.linspace(freq_vec[0], freq_vec[-1], int(n_bins*target_sr))
  xi, yi = np.meshgrid(time_vec, ures_freq_vec)
  X = interpn(points=(time_vec, freq_vec), values=X.T, xi=(xi, yi), method='splinef2d')

  X = dct(X, type=2, axis=0, norm='ortho')
  X = X[1:n_coef,:]
  for i in range(1, n_deltas+1):
    delt = librosa.feature.delta(X, order = i)
    X = np.concatenate([X, delt], axis=0)
  return time_vec, freq_vec, X