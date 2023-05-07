
from feature_extractor import FeatureExtractor
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
from utils import *

from scipy.signal import find_peaks
import matplotlib.pyplot as plt



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
    # for feature in features:
    #     end_point += int(sample_rate * 0.03)
    #     if feature < threshold:
    #         silence.append((start_point, end_point))
    #     else:
    #         vowel.append((start_point, end_point))
    #     start_point += int(sample_rate * 0.03)
    
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
    file_infos = {
        '30FTN': {'start': 0, 'end': 6.78},
        '42FQT': {'start': 0, 'end': 5.79},
        '44MTT': {'start': 0, 'end': 9.27},
        '45MDV': {'start': 0, 'end': 7.42}
    }
    
    file_lst = os.listdir("./TinHieuHuanLuyen")
    for file_name in file_infos.keys():
        file_path = os.path.join("./TinHieuHuanLuyen", file_name + ".wav")
        start = file_infos[file_name]['start']
        end = file_infos[file_name]['end']
        
        sample_rate, signal = read_wav_file(file_path)
        signal = normalization(signal)
        signal = filter(signal)

        lab_files = read_lab_file("./TinHieuHuanLuyen")

        segmented_values = split_audio("./TinHieuHuanLuyen", lab_files)
        
        window_size = 0.03  # kích thước cửa sổ là 30ms
        
        extractor = FeatureExtractor()
        segment = signal[int(start*sample_rate):int(end*sample_rate)]

        frames = split_frame(signal, sample_rate, window_size, start, end)
        
        ma_features = extractor.ma(frames, is_statistic=False)
        mean_ma_features = np.mean(ma_features)
        print(mean_ma_features)

        silence, vowel = define_frame_type(frames, ma_features, mean_ma_features)
        
        draw_point = get_data_for_draw(vowel)
        plot_voice_segment(signal, draw_point)
        spectrograms=[]
        for start, end in vowel:
            segment = signal[start:end]
            spectrograms.append(segment)
        
        
        data_array_spectrograms = np.vstack(spectrograms)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        f0_peak_searching = get_f0_peak_searching(data_array_spectrograms,sample_rate)
        
        for i in range(len(f0_peak_searching)):
            if f0_peak_searching[i] < 70 or f0_peak_searching[i] > 400:
                f0_peak_searching[i] = float("Nan")
      
        x = np.arange(int(signal.shape[0] / (sample_rate * 0.03)))
        y = np.full(shape = x.shape, fill_value=np.nan)
        count = 0
        for start, end in vowel:
            idx = int(start / (0.03 * sample_rate))
            y[idx] = f0_peak_searching[count]
            count += 1

        ax1.scatter(x, y)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("F0 (Hz)")
        ax1.set_title("F0 contour using peak searching")

    
        
        
        f0_peak_hps = get_f0_hps(data_array_spectrograms,sample_rate)
        for i in range(len(f0_peak_hps)):
            if f0_peak_hps[i] < 70 or f0_peak_hps[i] > 400:
                f0_peak_hps[i] = float("Nan")
        draw_point = get_data_for_draw(vowel)
        x = [((start+end)/2)/sample_rate for start, end in vowel]
        y = f0_peak_hps
        ax2.scatter(x, y)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("F0 (Hz)")
        ax2.set_title("F0 contour using harmonic product spectrum")
        
        f0_mean = np.nanmean(f0_peak_searching)
        print("Mean F0 (peak searching): ", f0_mean)

        f0_mean = np.nanmean(f0_peak_hps)
        print("Mean F0 (harmonic product spectrum): ", f0_mean)
        plt.tight_layout()
        plt.show()

        
       

        


