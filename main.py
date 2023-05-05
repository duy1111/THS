
from feature_extractor import FeatureExtractor
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import *

from scipy.io import wavfile

def read_wav_file(file_path: str):
    sample_rate, signal = wavfile.read(filename = file_path)
    return sample_rate, signal

def read_lab_file(directory: str):
    file_lst = os.listdir(directory)
    lab_files = [file for file in file_lst if file.endswith(".lab")]
    return lab_files

def split_audio(directory: str,lab_files : list):
    final_result = {}
    for file in lab_files:
        result = {"silence": [], "vowel": []}
        file_path = os.path.join(directory, file)
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                values = line.split("\t")
                values[-1] = values[-1].replace("\n", "")
                if len(values) > 2 and values[-1] == "sil":
                    result["silence"].append((float(values[0]), float(values[1])))
                elif len(values) >  2 and values[-1] != "sil":
                    result["vowel"].append((float(values[0]), float(values[1])))
        final_result[file] = result
    return final_result

def split_frame(signal: np.ndarray, sample_rate: float ,window_size: float, start: float, end: float):
    
    start_point = int(sample_rate * start)
    end_point = int(sample_rate * end)
    window_size = int(sample_rate * window_size)


    frames = []

    while start_point < end_point:
        frames.append(signal[start_point : start_point + window_size])
        start_point += window_size

    if start_point < end_point:
        frames.append(signal[start_point : end_point])

    return frames

def BinarySearch(start : float, end : float, start_list : list, end_list : list):
    middle = (start + end) / 2

    while middle > min(end_list) or middle < max(start_list):
        if middle > min(end_list):
            end = middle
        elif middle < max(start_list):
            start = middle
        
        middle = (start + end) / 2

    return middle




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
    frames = split_frame(signal, sample_rate, 0.03, start, end)
    ma_features = extractor.ma(frames, is_statistic=False)
    print(ma_features)


    
    silence, vowel = define_frame_type(frames, ma_features, 0.18)
    draw_point = get_data_for_draw(vowel)
    plot_voice_segment(signal, draw_point)
    
    
    


    


