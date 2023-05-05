
from feature_extractor import FeatureExtractor
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
from utils import *

from scipy.signal import find_peaks



def define_frame_type(frames: list, features:list, threshold:float):
    silence = []
    vowel = []
    start_point = 0
    end_point = 0
    for frame, feature in zip(frames, features):
        end_point += int(sample_rate * 0.03)
        if feature < threshold:
            silence.append((start_point, end_point))
        else:
            vowel.append((start_point, end_point))
        start_point += int(sample_rate * 0.03)
    return silence, vowel

def get_data_for_draw(vowel:list):
    result_list = []
    start_draw = vowel[0][0]
    for i in range(len(vowel) - 1):
       
        if vowel[i][1] == vowel[i + 1][0]:
            end_draw = vowel[i+1][1]
        else:
            result_list.append((start_draw,end_draw))
            start_draw = vowel[i+1][0]
        
        if i == len(vowel) - 2:
            result_list.append((start_draw, end_draw))
    return result_list


def plot_voice_segment(signal:np.ndarray,result_list: list):
    plt.plot(np.arange(len(signal)), signal)
    for value in result_list:
        plt.axvline(value[0], color = "r")
        plt.axvline(value[1], color = "r")
    plt.show()

if __name__ == "__main__":
    window_size = 0.03  # kích thước cửa sổ là 30ms
    hop_size = 0.01  # khoảng cách giữa hai khung là 10ms
    start = 0.0
    end = 6.78
    file_lst = os.listdir("./TinHieuHuanLuyen")
    sample_rate, signal = read_wav_file("./TinHieuHuanLuyen/30FTN.wav")
    
    lab_files = read_lab_file("./TinHieuHuanLuyen")


    segmented_values = split_audio("./TinHieuHuanLuyen", lab_files)
    
    
    extractor = FeatureExtractor()
    segment = signal[int(start*sample_rate):int(end*sample_rate)]
    print(segment)
    frames = split_frame(signal, sample_rate, window_size, start, end)
    ma_features = extractor.ma(frames, is_statistic=False)
 
    

    
    silence, vowel = define_frame_type(frames, ma_features, 0.18)
    

    draw_point = get_data_for_draw(vowel)
    print(draw_point)
    plot_voice_segment(signal, draw_point)
    n_fft = 1323
    hop_length = 330
    
    spectrograms = []
    
    signal = signal.astype(np.float32)
  
    for start, end in vowel:
        segment = signal[start:end]
 
        spectrogram = librosa.core.stft(segment, n_fft=n_fft, hop_length=hop_length, window='hann', center=True, pad_mode='reflect')
        spectrogram= np.abs(spectrogram)
        

        frame_peaks, _ = find_peaks(spectrogram.max(axis=0), distance=5)
        spectrograms.append((spectrogram, frame_peaks))
    
    


    


