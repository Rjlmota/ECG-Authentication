import numpy as np
import pandas as pandas
from scipy.signal import find_peaks

class Extractor:


    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

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
        
        features['r_y'] = segment[peak_x]
        features['r_x'] = peak_x 
        
        
        
        search_window_size = int(sample_rate*0.5) # 500ms

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
        


        if (not np.isnan(valor_x)):
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
        


    def get_mean_and_std(self, feature_name, final_features):
        return np.nanmean(np.array(final_features[feature_name])), np.nanstd(np.array(final_features[feature_name]))
        
    def get_mean(self, feature_name, final_features):
        return np.nanmean(np.array(final_features[feature_name])) 

        
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
        mean_q, stdev_q = self.get_mean_and_std("q_y", final_features) #np.nanmean(np.array(final_features['q_y'])), np.nanstd(np.array(final_features['q_y'])) 
        mean_r, stdev_r = self.get_mean_and_std("r_y", final_features)#np.nanmean(np.array(final_features['r_y'])), np.nanstd(np.array(final_features['r_y']))
        mean_s, stdev_s = self.get_mean_and_std("s_y", final_features)#np.nanmean(np.array(final_features['s_y'])), np.nanstd(np.array(final_features['s_y']))
        mean_rq_amplitude = self.get_mean('rq_amplitude', final_features)#np.nanmean(np.array(final_features['rq_amplitude'])) 
        
        mean_qt_distance = self.get_mean('q_t_distance', final_features)#np.nanmean(np.array(final_features['q_t_distance']))#, np.nanstd(np.array(final_features['q_t_distance']))
        mean_qs_distance = self.get_mean('q_s_distance', final_features)#np.nanmean(np.array(final_features['q_s_distance']))#, np.nanstd(np.array(final_features['q_s_distance']))
        
        
        mean_p, stdev_p = self.get_mean_and_std('p_y', final_features) #np.nanmean(np.array(final_features['p_y'])), np.nanstd(np.array(final_features['p_y']))
        mean_t, stdev_t = self.get_mean_and_std('t_y', final_features)  #np.nanmean(np.array(final_features['t_y'])), np.nanstd(np.array(final_features['t_y']))
        

        
        mean_qrs_onset =  self.get_mean('qrs_onset_y', final_features) #np.nanmean(np.array(final_features['qrs_onset_y']))
        mean_qrs_offset = self.get_mean('qrs_offset_y', final_features)  #np.nanmean(np.array(final_features['qrs_offset_y']))


        mean_p_onset =  self.get_mean('p_onset_y', final_features) #np.nanmean(np.array(final_features['p_onset_y']))
        mean_p_offset =  self.get_mean('p_offset_y', final_features) #np.nanmean(np.array(final_features['p_offset_y']))


        mean_t_onset = self.get_mean('t_onset_y', final_features) #np.nanmean(np.array(final_features['t_onset_y']))
        mean_t_offset = self.get_mean('t_offset_y', final_features)#np.nanmean(np.array(final_features['t_offset_y']))

        mean_qrs_interval = self.get_mean('qrs_interval', final_features) #np.nanmean(np.array(final_features['qrs_interval']))#, np.nanstd(np.array(final_features['qrs_interval']))
        

        list_of_tp_segment = []
        for i in range(len(np.array(final_features['t_offset_x']))):
            if(i+1 < len(np.array(final_features['p_onset_x']))):
                list_of_tp_segment.append(abs(final_features['t_offset_x'][i] - final_features['p_onset_x'][i+1]))

        mean_tp_segment = np.nanmean(np.array(list_of_tp_segment)) ##NEW FEATURE

        mean_qt_interval = self.get_mean('q_t_interval', final_features) #np.nanmean(np.array(final_features['q_t_interval'])) ## NEW FEATURE

        mean_st_interval = self.get_mean('s_t_interval', final_features)#np.nanmean(np.array(final_features['s_t_interval'])) ## NEW FEATURE

        mean_t_wave = self.get_mean('t_wave', final_features) #np.nanmean(np.array(final_features['t_wave'])) ## NEW FEATURE

        mean_pq_segment = self.get_mean('pq_segment', final_features) #np.nanmean(np.array(final_features['pq_segment'])) ## NEW FEATURE


        mean_st_segment = self.get_mean('s_t_segment', final_features) #np.nanmean(np.array(final_features['s_t_segment'])) ## NEW FEATURE

       # mean_tp_segment = self.get_mean('tp_segment', final_features)#np.nanmean(np.array(final_features['tp_segment'])) ## NEW FEATURE



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
