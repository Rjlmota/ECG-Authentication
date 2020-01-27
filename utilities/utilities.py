import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, lfilter
from scipy.signal import butter, lfilter
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
# TODO:
# 1 - Escrever as docstrings de todos os métodos e classes pra tentar substituir oscoentários
# 2- Passar pra inglês os comentários restantes


## Function that automatically detects the signal's sample rate
def detect_sample_rate(signal):
    start_time = 2
    time_window = 1
    segment = signal[(signal['time'] >= start_time) & (signal['time'] < start_time + time_window)]
    return len(segment)



## To-do comment
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



## Uma classe que implementa o método de filtragem (para usar no pipeline)
## A class that implements the filter method (for pipeline usage).

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


    
## Uma classe que implementa a extração de segmentos (para usar no pipeline)
## A class that impelments the feature extraction of segments (for pipeline usage)

    
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, number_of_segments, sample_rate):
        self.number_of_segments = number_of_segments
        self.sample_rate = sample_rate
        
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        self.X = X
        return self.segmentate()
        

        
    def random_segment_generator(self): 
        window = self.sample_rate*2
        # limit for iteration
        cap = len(self.X) - (window)-1
        
        initial_index = np.random.randint(cap)
        # index maior
        final_index = initial_index + window
        # pegar a janela aleatoria
        return np.array(self.X[initial_index:final_index])
        
        
        
    def segmentate(self):

        # armazenará as features extraídas
        
        columns_names = ['mean_q', 'mean_r', 'mean_s',
                         'stdev_q', 'stdev_r','stdev_s',
                         'mean_qrs_interval', 'mean_rr_interval',
                         'mean_rq_amplitude'] # peaks number as a feature later?
    
        self.features_df = pd.DataFrame(columns=columns_names)
        
        
        # pega todos os segmentos originais possíveis de se obter com os dados originais
        # e extrai as features de cada segmento desses
        
        # Sample rate indicates how many data points we have for 1second of measeurement. 
        # We want 2 seconds, so we multiply by 2.
        step = self.sample_rate * 2 
        
        # extraindo todos os segmentos possíveis com passo pré definido.
        for time_window in range(step, len(self.X), step):
            segment = self.X[time_window-step:time_window]

            extracted_features = self.extract_features(segment, self.sample_rate)
            
            ## adiciona as features recentemente extraídas ao dataframe final.
            if not (np.any(extracted_features)):
                continue
            self.features_df.loc[len(self.features_df)] = extracted_features     


        #print(f"Generated {len(self.features_df)} original segments")        

        original_segments_only = self.features_df.iloc[int(len(self.features_df)/2):]
        

        # 1° caso: já existem segmentos suficiente, basta escolher n segmentos aleatórios.

        if(len(self.features_df) >= self.number_of_segments):
            correct_sized_df = self.features_df.sample(n=self.number_of_segments, random_state=1) 
            original_sample = correct_sized_df[:int(len(self.correct_sized_df)/2)] 
            correct_sized_df = correct_sized_df[int(len(self.correct_sized_df)/2):]
            #print(type(correct_sized_df), type(original_sample))
            
            return correct_sized_df, original_sample
        
        #remaining_data = self.features_df.iloc[:int(len(self.features_df)/2)]
        remaining_data = pd.DataFrame(columns=columns_names)    
        
        # 2° caso: Não há segmentos suficientes, necessário gerar segmentos aleatórios
        
        # variáveis de controle para impedir um loop infinito
        max_attempt = 10_000
        current_attempt = 0
        
        
        # gera os segmentos aleatórios, e extrai as features de cada segmento desses
        while(len(remaining_data) < self.number_of_segments):
           
            #checa máximo de iterações
            if(current_attempt >= max_attempt):
                break
            
            # gera o segmento aleatóprio aqui
            random_segment = self.random_segment_generator()
            
            # extrai features do segmento aleatório
            extracted_features = np.array(self.extract_features(random_segment, self.sample_rate))
            # adiciona um ao contador de iterações
            current_attempt += 1
            

            if not (np.any(extracted_features)):
                continue
            elif( (remaining_data == extracted_features).all(1).any()):
                #print("ELIMINADO POR SER CÓPIA")
                continue
            else:
                # adiciona o segmento aeatório no dataframe final
                remaining_data.loc[len(remaining_data)] = extracted_features
        
        return remaining_data, original_segments_only

        
    def get_peaks(self, segment, sample_rate):
        #obtém o valor mais alto presente no segmento.
        max_value = np.max(np.array(segment).astype(float)) # renato preguiçoso se tu não ver isso esse comentário vai ficar aqui
                                                                # denegrindo sua imagem
            
        peaks, _ = find_peaks(segment, height=0.6*max_value, distance=sample_rate*0.2)

        if(len(peaks) < 1):
            return []

        if(peaks[-1] + sample_rate*0.040 > len(segment)):
            peaks = np.delete(peaks, [len(peaks)-1])

        if(len(peaks) < 1):
            return []
        if(peaks[0] -sample_rate*0.040 < 0):
            peaks = np.delete(peaks, [0])

        if(len(peaks) < 1):
            return []

        return peaks
        
        
    def extract_features(self, segment, sample_rate):
        '''
        Extracts the peak's indexes based on a time distance based on the predefined sample rate.

        If there's not enough data to get the number of segments needed, this implementation
        will gengerate random segments from the original data

        Returns:
        features list containing the following information:

        (meanQ, meanR, mmeanS, deltaR, stdevQ, stdevR, stdevS
        qrs_interval, rr_interval, rq_amplitude) + number_of_peaks? 
        '''

        # initializing variables.
        q_list = []
        s_list = []
        r_list = []
        qrs_interval_list = []
        rq_amplitude_list = []
        features = []

        peaks = self.get_peaks(segment, sample_rate)
        if(len(peaks) < 1):
            return np.array([])

        #finding difference between r times
        difference_between_r = np.diff(peaks) 

        for i in range(len(difference_between_r)):
            if difference_between_r[i] < (sample_rate*0.12):
                #dropped due to this article Weighted Conditional Random Fields for Supervised Interpatient Heartbeat Classification
                difference_between_r.pop(i)

        if np.mean(difference_between_r) == np.nan:
            difference_between_r = 0
        
        #interval in ms corresponding to each dataset's time unit
        mean_rr_interval = np.mean(difference_between_r)*(1000.0/float(self.sample_rate)) 

        for peak in peaks:
            # this search windows surround the R peak found before by 40ms,
            # which is enough for detecting the said features
            step = int(sample_rate*0.040)
            search_window = segment[peak-step:peak+step]
            half_index = int(len(search_window)//2)


            # finding q:
            q_value = min(search_window[:half_index])
            q_list.append(q_value)
            q_instant_time = search_window[:half_index].argmin()


            # finding s:
            s_value = min(search_window[half_index:])
            s_list.append(s_value)
            s_instant_time = search_window[half_index:].argmin()

            #defing qrs interval
            single_qrs_interval = ( (s_instant_time+step) - q_instant_time) * (1000.0/float(sample_rate))
            qrs_interval_list.append(single_qrs_interval)


            #obtaining rq_amplitude
            single_rq_amplitude = abs(segment[peak] - q_value)
            rq_amplitude_list.append(single_rq_amplitude)


        r_list = [segment[index] for index in peaks]

        mean_q = np.mean(q_list)
        mean_s = np.mean(s_list)
        mean_r = np.mean(r_list)

        stdev_q = np.std(q_list)
        stdev_s = np.std(s_list)
        stdev_r = np.std(r_list)

        mean_rq_amplitude = np.mean(rq_amplitude_list)
        mean_qrs_interval = np.mean(qrs_interval_list)


        diff_r = np.diff(peaks)
        deltaR = 0 if(len(diff_r) == 0) else np.mean(diff_r)
        

        ff = np.array([mean_q, mean_r, mean_s, stdev_q,
                stdev_r, stdev_s, mean_rr_interval,
                mean_rq_amplitude, mean_qrs_interval])
        
        #print(peaks)
        #print(ff)
        
        return ff