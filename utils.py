import os
import numpy as np
from scipy.io import wavfile

import scipy.signal as signal
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





def get_f0_peak_searching(ma_features: np.ndarray, sample_rate: int) -> tuple:
    f0_list = []
    time_list = []
    for i in range(ma_features.shape[0]):
        
        acf = signal.correlate(ma_features[i], ma_features[i], mode="full")
        acf = acf[len(acf)//2:]
        peaks, _ = signal.find_peaks(acf, distance=10)
        if len(peaks) > 0:
            peak = peaks[0]
            f0 = sample_rate / peak
            f0_list.append(f0)
        else:
            f0_list.append(0)
        time_list.append(i * 0.03)
    return f0_list, time_list

def get_f0_hps(ma_features: np.ndarray, sample_rate: int) -> tuple:
    f0_list = []
    time_list = []
    for i in range(ma_features.shape[0]):
        signal = ma_features[i]
        signal -= np.mean(signal)
        N = len(signal)
        hps_signal = signal[:N//2] * signal[:N//2//2] * signal[:N//2//3] * signal[:N//2//4]
        fft_signal = np.abs(np.fft.rfft(hps_signal))
        freqs = np.fft.rfftfreq(N, d=1/sample_rate)
        peak = freqs[np.argmax(fft_signal)]
        f0_list.append(peak)
        time_list.append(i * 0.03)
    return f0_list, time_list