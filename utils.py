import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

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
   
    
    for i in range(ma_features.shape[0]):
        mag_spectrum = np.abs(np.fft.fft(ma_features[i]))[:len(ma_features[i]) // 2]
        freqs = np.fft.fftfreq(len(mag_spectrum), d=1/sample_rate)[:len(mag_spectrum)]
        peaks, _ = find_peaks(mag_spectrum, distance=5)
        
        if len(peaks) > 0:
            peak_freq = freqs[peaks[0]]
            f0 = sample_rate / peak_freq
            f0_list.append(f0)
        else:
            f0_list.append(0)     
    
    return f0_list
   
   
def get_f0_hps(ma_features: np.ndarray, sample_rate: int) -> tuple:
    f0_list = []
 
    
    for i in range(ma_features.shape[0]):
        mag_spectrum = np.abs(np.fft.fft(ma_features[i]))[:len(ma_features[i]) // 2]
        freqs = np.fft.fftfreq(len(mag_spectrum), d=1/sample_rate)[:len(mag_spectrum)]
        
        hps_spectrum = mag_spectrum.copy()
        
        for harmonic in range(2, 8): 
            if len(mag_spectrum) % harmonic == 0:
                hps_spectrum *= mag_spectrum[::harmonic]
            else:
                repeated_mag_spectrum = np.repeat(mag_spectrum[::harmonic], harmonic)[:len(mag_spectrum)]
                hps_spectrum *= repeated_mag_spectrum

        
        peak_idx = np.argmax(hps_spectrum)
        peak_freq = freqs[peak_idx]
        if peak_freq == 0:
            f0 = 0
        else:
            f0 = sample_rate / peak_freq
        f0_list.append(f0)
   
    
    return f0_list
