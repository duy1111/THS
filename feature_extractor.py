import numpy as np
import librosa

class FeatureExtractor:
    def __init__(self):
        pass
    
    def signum(self, frame):
        result = np.zeros(shape=(len(frame)))
        for i in range(len(frame)):
            if (frame[i] >= 0): 
                result[i] = 1
            else: 
                result[i] = -1
        
        return result

    def ste(self, frame_list: list):
        # tính STE bằng cách bình phương và tính trung bình các giá trị trong khung
        result = []
        for frame in frame_list:
            result.append(np.sum(np.multiply(frame,frame)))
        return np.mean(np.array(result) / np.max(np.abs(result)))   
    
    def zcr(self, frame_list: list):
        result = []
        for frame in frame_list:
            result.append(np.sum(np.abs(self.signum(frame[1:]) - self.signum(frame[0 : len(frame) - 1]))))
        return np.mean(np.array(result) / np.max(np.abs(result)))
    
    def ma(self, frame_list: list, is_statistic: bool):
        result = []
        for frame in frame_list:
            result.append(np.mean(np.abs(frame)))
        if is_statistic:
            return np.mean(np.array(result) / np.max(np.abs(result)))
        return np.array(result) / np.max(np.array(result))

    def extract(self, signal, sample_rate):
        frame_length = int(0.02 * sample_rate)  # frame length: 20ms
        frame_shift = int(0.01 * sample_rate)  # frame shift: 10ms
        frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=frame_shift).T
        
        ma_feats = self.ma(frames, is_statistic=True)  # calculate MA features
        f0 = None
        spectrum = None
        if len(signal) > 0:
            spectrum = np.abs(np.fft.fft(signal))**2  # calculate spectrum using FFT
            freqs = np.fft.fftfreq(len(spectrum), 1/sample_rate)  # calculate corresponding frequency values
            idx = np.argmax(spectrum[:len(spectrum)//2])  # find index of maximum peak in spectrum
           



