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
        

        
    def random_segment_generator(self, time_seconds=3): 
        window = self.sample_rate*time_seconds#2
        # limit for iteration
        cap = len(self.X) - (window)-1
        
        initial_index = np.random.randint(cap)
        # index maior
        final_index = initial_index + window
        # pegar a janela aleatoria
        return np.array(self.X[initial_index:final_index])
        

    def segmentate(self):

        # armazenará as features extraídas
        
        columns_names = ['mean_q', 'mean_r', 'mean_s', 'mean_p', 'mean_t',
                         'stdev_q', 'stdev_r','stdev_s',
                         'mean_rr_interval', 'mean_rq_amplitude', 'mean_qrs_interval',
                         'mean_qs_distance', 'mean_qt_distance', 'mean_qrs_offset', 'mean_qrs_onset',
                         'mean_p_onset', 'mean_p_offset', 'mean_t_onset', 'mean_t_offset',
                         'mean_qt_interval', 'mean_st_interval', 'mean_t_wave', 'mean_pq_segment',
                         'mean_st_segment', 'mean_tp_segment', 'mean_pp_interval'
                         ] # peaks number as a feature later?
    


 



        self.features_df = pd.DataFrame(columns=columns_names)
        
        
        # pega todos os segmentos originais possíveis de se obter com os dados originais
        # e extrai as features de cada segmento desses
        
        # Sample rate indicates how many data points we have for 1second of measeurement. 
        # We want 2 seconds, so we multiply by the amount of seconds wanted.
        time_seconds = 3
        step = self.sample_rate * time_seconds 
        
        # extraindo todos os segmentos possíveis com passo pré definido.
        for time_window in range(step, len(self.X), step):
            segment = self.X[time_window-step:time_window]

            extracted_features = self.extract_features(segment, self.sample_rate)
            
            ## adiciona as features recentemente extraídas ao dataframe final.
            if not (np.any(extracted_features)):
                continue

            #print(len(self.features_df.columns))
            #print(len(extracted_features))

            self.features_df.loc[len(self.features_df)] = extracted_features            
        

        # 1° caso: já existem segmentos suficiente, basta escolher n segmentos aleatórios.


        if(len(self.features_df) >= self.number_of_segments):
            correct_sized_df = self.features_df.sample(n=self.number_of_segments, random_state=1) 
            #print(f"I selected {len(original_sample)} samples")
            return correct_sized_df
        


        original_segments_only = self.features_df.copy()
        
        # 2° caso: Não há segmentos suficientes, necessário gerar segmentos aleatórios
        
        # variáveis de controle para impedir um loop infinito
        max_attempt = 10_000
        current_attempt = 0
        
        
        # gera os segmentos aleatórios, e extrai as features de cada segmento desses
        while(len(self.features_df) < self.number_of_segments):
           
            #checa máximo de iterações
            if(current_attempt >= max_attempt):
                break
            # adiciona um ao contador de iterações
            current_attempt += 1
            
            # gera o segmento aleatóprio aqui
            random_segment = self.random_segment_generator()
            
            # extrai features do segmento aleatório
            extracted_features = np.array(self.extract_features(random_segment, self.sample_rate))

            
            # checa se é nulo
            if not (np.any(extracted_features)):
                continue

            elif( (self.features_df == extracted_features).all(1).any()):
                #print("ELIMINADO POR SER CÓPIA")
                continue
            else:
                # adiciona o segmento aeatório no dataframe final
                self.features_df.loc[len(self.features_df)] = extracted_features
        

        #print(f"I generated {len(self.features_df)} samples for training")

        return original_segments_only, self.features_df

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
        
    

    def find_offset(self, feature_x_position, feature_y_value, segment):
        time_window = int(self.sample_rate*0.04) #40ms
        offset_y_value = float('-inf')
        
        offset_x_position = float('NaN')
        
        max_walk = feature_x_position + time_window
        if(max_walk > len(segment)):
           max_walk = len(segment)
           
        for x in range(feature_x_position, max_walk):
           candidate = abs(feature_y_value - segment[x]) / abs(feature_x_position - x)
        
           if(candidate > offset_y_value):
               offset_y_value = candidate
               offset_x_position = x
           
        
        return offset_x_position


    def find_onset(self, feature_x_position, feature_y_value, segment):
        time_window = int(self.sample_rate*0.04) #40ms
        onset_y_value = float('-inf')
        
        onset_x_position = float('NaN')
        
        max_walk = feature_x_position - time_window
        if(max_walk < 0):
           max_walk = 0
           
        for x in range(feature_x_position, max_walk, -1):
           candidate = abs(feature_y_value - segment[x]) / abs(feature_x_position - x)
        
           if(candidate > onset_y_value):
               onset_y_value = candidate
               onset_x_position = x
           
        
        return onset_x_position


    def extract_local_features(self, peak_x, segment, sample_rate):
        #y - voltage
        #x - time index (depends on sample rate!)
        
        features = {}
        
        #peak_y = segment[peak_x] #gets the peak value
        features['r_y'] = segment[peak_x]
        features['r_x'] = peak_x 
        
        
        
        search_window_size = int(sample_rate*0.5) # 500ms
        #get P_Q(x, y)
        
        #local_peaks_left = [float('-inf'), float('inf')] # [higher_peak,lower_peak]
        local_max = [float('-inf'), 0] # = [higher_value, index]
        local_min = [float('inf'), 0] # = [minimum value, index]
        for x in range(peak_x, (peak_x-search_window_size), -1):
            if (segment[x] > local_max[0]):
                local_max = [segment[x], x]
            
            if (segment[x] < local_min[0]):
                local_min = [segment[x], x]
        
        
        features['q_y'], features['q_x'] = local_min
        
        features['p_y'], features['p_x'] = local_max


        valor_x = self.find_offset(features['p_x'], features['p_y'], segment)
        


        if (not np.isnan(valor_x)):
            valor_y = segment[valor_x] 
        else: 
            valor_y = float('NaN')

        features['p_offset_y'] = valor_y
        features['p_offset_x'] = valor_x



        valor_x = self.find_onset(features['p_x'], features['p_y'], segment)
        


        if (not np.isnan(valor_x)):
            valor_y = segment[valor_x] 
        else: 
            valor_y = float('NaN')

        features['p_onset_y'] = valor_y
        features['p_onset_x'] = valor_x



        
        
        #local_peaks_left = [float('-inf'), float('inf')] # [higher_peak,lower_peak]
        local_max = [float('-inf'), 0] # = [higher_value, index]
        local_min = [float('inf'), 0] # = [minimum value, index]
        

        max_walk = (peak_x+search_window_size) if (peak_x+search_window_size) < len(segment) else len(segment) 

        for x in range(peak_x, max_walk):
            #print(x)
            if (segment[x] > local_max[0]):
                local_max = [segment[x], x]
            
            if (segment[x] < local_min[0]):
                local_min = [segment[x], x]
                
        
        features['s_y'], features['s_x'] = local_min
        
        features['t_y'], features['t_x'] = local_max
        

        valor_x = self.find_offset(features['t_x'], features['t_y'], segment)
        


        if (not np.isnan(valor_x)):
            valor_y = segment[valor_x] 
        else: 
            valor_y = float('NaN')

        features['t_offset_y'] = valor_y
        features['t_offset_x'] = valor_x
            
        


        valor_x = self.find_onset(features['t_x'], features['t_y'], segment)
        


        if (not np.isnan(valor_x)):
            valor_y = segment[valor_x] 
        else: 
            valor_y = float('NaN')

        features['t_onset_y'] = valor_y
        features['t_onset_x'] = valor_x
            


        valor_x = self.find_offset(features['s_x'], features['s_y'], segment)
        


        if (not np.isnan(valor_x)):
            valor_y = segment[valor_x] 
        else: 
            valor_y = float('NaN')

        features['qrs_offset_y'] = valor_y
        features['qrs_offset_x'] = valor_x

        

            valor_y = segment[valor_x] 
        else:
            valor_y = float('NaN')
        features['qrs_onset_y'] = valor_y
        features['qrs_onset_x'] = valor_x




        if(features['qrs_onset_y'] == float('-inf')):
            features['qrs_onset_y'] = float('NaN')

        if(features['qrs_offset_y'] == float('-inf')):
            features['qrs_offset_y'] = float('NaN')
                
                
        features['qrs_interval'] = features['qrs_offset_x'] - features['qrs_onset_x']
        
        features['rq_amplitude'] = abs(segment[peak_x] - features['q_y'])
        
        features['q_t_distance'] = abs(features['t_x'] - features['q_x'])
        
        features['q_t_interval'] = abs(features['qrs_onset_x'] - features['t_offset_x'])




        features['s_t_segment'] = abs(features['qrs_offset_x'] - features['t_onset_x'])

        features['s_t_interval'] = abs(features['qrs_offset_x'] - features['t_offset_x'])

        features['q_s_distance'] = abs(features['s_x'] - features['q_x'])

        features['t_wave'] = abs(features['t_offset_x'] - features['t_onset_x'])

        features['pq_segment'] = abs(features['p_offset_x'] - features['qrs_onset_x'])



        
        return features
        
        
        

        
    def extract_features(self, segment, sample_rate):
        peaks  = self.get_peaks(segment, sample_rate)
        
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
        mean_rr_interval = np.mean(difference_between_r)*(1000.0/float(sample_rate)) 
        
        features_global = []
        for peak in peaks:
            #print(peak)
            features_local = (self.extract_local_features(peak, segment, sample_rate))
            features_global.append(features_local)
            
        
        # juntando os dicionarios de features em um único só
        final_features = {}
        for key in features_global[0].keys():
            final_features[key] = tuple(diict[key] for diict in features_global)
        
        # obetendo as últimas features restantes
        mean_q, stdev_q = np.nanmean(np.array(final_features['q_y'])), np.nanstd(np.array(final_features['q_y'])) 
        mean_r, stdev_r = np.nanmean(np.array(final_features['r_y'])), np.nanstd(np.array(final_features['r_y']))
        mean_s, stdev_s = np.nanmean(np.array(final_features['s_y'])), np.nanstd(np.array(final_features['s_y']))
        mean_rq_amplitude = np.nanmean(np.array(final_features['rq_amplitude'])) 
        
        mean_qt_distance = np.nanmean(np.array(final_features['q_t_distance']))#, np.nanstd(np.array(final_features['q_t_distance']))
        mean_qs_distance = np.nanmean(np.array(final_features['q_s_distance']))#, np.nanstd(np.array(final_features['q_s_distance']))
        
        
        mean_p, stdev_p = np.nanmean(np.array(final_features['p_y'])), np.nanstd(np.array(final_features['p_y']))
        mean_t, stdev_t = np.nanmean(np.array(final_features['t_y'])), np.nanstd(np.array(final_features['t_y']))
        

        
        mean_qrs_onset = np.nanmean(np.array(final_features['qrs_onset_y']))
        mean_qrs_offset = np.nanmean(np.array(final_features['qrs_offset_y']))


        mean_p_onset =  np.nanmean(np.array(final_features['p_onset_y']))
        mean_p_offset =  np.nanmean(np.array(final_features['p_offset_y']))


        mean_t_onset = np.nanmean(np.array(final_features['t_onset_y']))
        mean_t_offset = np.nanmean(np.array(final_features['t_offset_y']))

        mean_qrs_interval = np.nanmean(np.array(final_features['qrs_interval']))#, np.nanstd(np.array(final_features['qrs_interval']))
        




        list_of_tp_segment = []
        for i in range(len(np.array(final_features['t_offset_x']))):
            if(i+1 < len(np.array(final_features['p_onset_x']))):
                list_of_tp_segment.append(abs(final_features['t_offset_x'][i] - final_features['p_onset_x'][i+1]))

        mean_tp_segment = np.nanmean(np.array(list_of_tp_segment)) ##NEW FEATURE





        mean_qt_interval = np.nanmean(np.array(final_features['q_t_interval'])) ## NEW FEATURE

        mean_st_interval = np.nanmean(np.array(final_features['s_t_interval'])) ## NEW FEATURE

        mean_t_wave = np.nanmean(np.array(final_features['t_wave'])) ## NEW FEATURE

        mean_pq_segment = np.nanmean(np.array(final_features['pq_segment'])) ## NEW FEATURE


        mean_st_segment = np.nanmean(np.array(final_features['s_t_segment'])) ## NEW FEATURE


        difference_between_p =  np.diff(np.array(final_features['p_x']))
        mean_pp_interval = np.mean(difference_between_p)*(1000.0/float(sample_rate)) ##NEW FEATURE



            
 
        ff = np.array([mean_q, mean_r, mean_s, mean_p, mean_t,
                      stdev_q, stdev_r, stdev_s, 
                      mean_rr_interval, mean_rq_amplitude, mean_qrs_interval,
                      mean_qs_distance, mean_qt_distance, mean_qrs_offset, mean_qrs_onset,
                      mean_p_onset, mean_p_offset, mean_t_onset, mean_t_offset,
                      mean_qt_interval, mean_st_interval, mean_t_wave, mean_pq_segment,
                      mean_st_segment, mean_tp_segment, mean_pp_interval])     
        
        
        
        return ff
 