
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




if __name__ == "__main__":
    file_infos = {
        '01MDA': {'start': 0.00, 'end': 5.58,'F0mean':135.5,'F0std':5.4 },
        '02FVA': {'start': 0.00, 'end': 7.18,'F0mean':239.7,'F0std':5.6},
        '03MAB': {'start': 0.00, 'end': 9.37,'F0mean':115.0,'F0std':4.5},
        '06FTB': {'start': 0.00, 'end': 12.75,'F0mean':202.9,'F0std':15.5}
    }
    
    
    file_lst = os.listdir("./TinHieuKiemThu")
    for file_name in file_infos.keys():
        print(file_name)
        file_path = os.path.join("./TinHieuKiemThu", file_name + ".wav")
        start = file_infos[file_name]['start']
        end = file_infos[file_name]['end']
        
        sample_rate, signal = read_wav_file(file_path)
        signal = normalization(signal)
        signal = filter(signal)

        lab_files = read_lab_file("./TinHieuKiemThu")

        segmented_values = split_audio("./TinHieuKiemThu", lab_files)
        
        window_size = 0.03  # kích thước cửa sổ là 30ms
        
        extractor = FeatureExtractor()
        segment = signal[int(start*sample_rate):int(end*sample_rate)]

        frames = split_frame(signal, sample_rate, window_size, start, end)
        
        ma_features = extractor.ma(frames, is_statistic=False)
        mean_ma_features = np.mean(ma_features)
        print(mean_ma_features)
       
        silence, vowel = define_frame_type(frames, ma_features, 0.18163196749981997)
        fig = plt.figure()
        axs = fig.subplots(nrows=3)
        
        draw_point = get_data_for_draw(vowel)
        axs[0].plot(np.arange(len(signal)), signal)
        
        
        for value in draw_point:
            axs[0].axvline(value[0], color = "r")
            axs[0].axvline(value[1], color = "r")
        

        spectrograms=[]
        for start, end in vowel:
            segment = signal[start:end]
            spectrograms.append(segment)
        
        
        data_array_spectrograms = np.vstack(spectrograms)

        f0_peak_searching = get_f0_peak_searching(data_array_spectrograms,sample_rate)
        
        for i in range(len(f0_peak_searching)):
            if f0_peak_searching[i] < 70 or f0_peak_searching[i] > 400:
                f0_peak_searching[i] = float("Nan")
      
        f0_mean = np.nanmean(f0_peak_searching)
        for i in range(len(f0_peak_searching)):
            if f0_peak_searching[i] < (0.6 *f0_mean)  or f0_peak_searching[i] > (1.5 *f0_mean):
                f0_peak_searching[i] = float("Nan")

        f0_mean = np.nanmean(f0_peak_searching)
        
        dental_mean_f0 = np.abs(file_infos[file_name]['F0mean'] - f0_mean)
        std = np.nanstd(f0_peak_searching)
        dental_std_f0 = np.abs(file_infos[file_name]['F0std'] - std)

        x = np.arange(int(signal.shape[0] / (sample_rate * 0.03)))
        y = np.full(shape = x.shape, fill_value=np.nan)
        count = 0
        for start, end in vowel:
            idx = int(start / (0.03 * sample_rate))
            y[idx] = f0_peak_searching[count]
            count += 1

        axs[1].scatter(x, y)
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("F0 (Hz)")
        axs[1].set_title(f"F0 contour using peak searching \n Mean F0 {f0_mean} std {std} \n F0 mean {file_infos[file_name]['F0mean']} F0 std {file_infos[file_name]['F0std']}")

    
        
        
        f0_peak_hps = get_f0_hps(data_array_spectrograms,sample_rate)
        for i in range(len(f0_peak_hps)):
            if f0_peak_hps[i] < 70 or f0_peak_hps[i] > 400:
                f0_peak_hps[i] = float("Nan")
        draw_point = get_data_for_draw(vowel)
        x = [((start+end)/2)/sample_rate for start, end in vowel]
        y = f0_peak_hps
        axs[2].scatter(x, y)
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("F0 (Hz)")
        axs[2].set_title("F0 contour using harmonic product spectrum")
        
        plt.tight_layout()
        plt.show()

        
       

        


