import numpy as np

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




