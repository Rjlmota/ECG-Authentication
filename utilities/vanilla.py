from sklearn.pipeline import Pipeline
from scipy.signal import butter, lfilter
from sklearn.base import BaseEstimator, TransformerMixin

#this function detects sample rate by counting how many measurements are present in one second. 
#it only works if the time signature is in seconds.
def detect_sample_rate(signal):
    start_time = 2
    time_window = 1
    segment = signal[(signal['time'] >= start_time) & (signal['time'] < start_time + time_window)]
    return len(segment)


#this is a external function originated from [link] and it works as part of the filter.
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

#this is the filter itself
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#The class Filter works as a interface to using the filter functions above. 
#The fit and transform model was made to fit in a pipeline fashion.
class Filter(BaseEstimator, TransformerMixin):
    def __init__(self, sample_rate, low_cut, high_cut, order = 5):
        self.sample_rate = sample_rate
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.order = order
        
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        self.X = X
        return butter_bandpass_filter(self.X, self.low_cut, self.high_cut, self.sample_rate, self.order)



