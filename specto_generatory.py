import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt


test_file = "audio_files/never-gonna-give-you-up.mp3"

test, sr = librosa.load(test_file)

FRAME_SIZE = 2048
HOP_SIZE = 512

S_scale = librosa.stft(test, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
print(S_scale.shape)
print(type(S_scale[0][0]))

Y_scale = np.abs(S_scale) ** 2
print(Y_scale.shape)
print(type(Y_scale[0][0]))

def plot_spectogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.show()

def plot_mel_spectogram(filter_banks, sr, x_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(filter_banks, sr=sr, x_axis="time")
    plt.colorbar(format="%+2.f")
    plt.show()
#plot_spectogram(Y_scale, sr, HOP_SIZE)

Y_log_scale = librosa.power_to_db(Y_scale)
plot_spectogram(Y_log_scale, sr, HOP_SIZE)
filter_banks = librosa.filters.mel(n_fft=FRAME_SIZE, sr=22050, n_mels=10)
plot_mel_spectogram(filter_banks, sr)