from utilities.vanilla import *
from utilities.extractor import Extractor
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin



# Uma classe que implementa a extração de segmentos (para usar no pipeline)
## A class that impelments the feature extraction of segments (for pipeline usage)

    
class Segmentator(BaseEstimator, TransformerMixin):


    def __init__(self, number_of_segments, sample_rate, segment_duration=3):
        self.number_of_segments = number_of_segments
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.original_segments_only = pd.DataFrame()
        columns_names = ['mean_q', 'mean_r', 'mean_s', 'mean_p', 'mean_t',
                         'stdev_q', 'stdev_r','stdev_s',
                         'mean_rr_interval', 'mean_rq_amplitude', 'mean_qrs_interval',
                         'mean_qs_distance', 'mean_qt_distance', 'mean_qrs_offset', 'mean_qrs_onset',
                         'mean_p_onset', 'mean_p_offset', 'mean_t_onset', 'mean_t_offset',
                         'mean_qt_interval', 'mean_st_interval', 'mean_t_wave', 'mean_pq_segment',
                         'mean_st_segment', 'mean_tp_segment', 'mean_pp_interval'
                         ] # peaks number as a feature later?
    

        self.features_df = pd.DataFrame(columns=columns_names)
        
        self.extractor = Extractor(self.sample_rate)


        
    def fit(self, DATA, y=None):
        return self
    
    def transform(self, DATA, y=None):
        self.DATA = DATA
        return self.segmentate()
        

        
    def random_segment_generator(self): 
        window = self.sample_rate*self.segment_duration #2
        # limit for iteration
        cap = len(self.DATA) - (window)-1
        
        initial_index = np.random.randint(cap)
        # index maior
        final_index = initial_index + window
        # pegar a janela aleatoria
        return np.array(self.DATA[initial_index:final_index])
        


    def detect_linear_segments(self):
        # pega todos os segmentos originais possíveis de se obter com os dados originais
        # e extrai as features de cada segmento desses
        
        # Sample rate indicates how many data points we have for 1second of measeurement. 
        # We want 2 seconds, so we multiply by the amount of seconds wanted.
        step = self.sample_rate * self.segment_duration 
        
        # extraindo todos os segmentos possíveis com passo pré definido.
        for time_window in range(step, len(self.DATA), step):
            segment = self.DATA[time_window-step:time_window]

            extracted_features = self.extractor.extract_features(segment, self.sample_rate)
            
            ## adiciona as features recentemente extraídas ao dataframe final.
            if not (np.any(extracted_features)):
                continue

            #print(len(self.features_df.columns))
            #print(len(extracted_features))

            self.features_df.loc[len(self.features_df)] = extracted_features   

    def segmentate(self):

        # extract segments in a linear fashion, sliding a time window defined in the variable "segment_duration"
        self.detect_linear_segments()
        # 1° caso: já existem segmentos suficiente, basta escolher n segmentos aleatórios.

        if(len(self.features_df) >= self.number_of_segments):
            correct_sized_df = self.features_df.sample(n=self.number_of_segments, random_state=1) 
            #print(f"I selected {len(original_sample)} samples")
            return correct_sized_df
        


        self.original_segments_only = self.features_df.copy()
        
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
            extracted_features = np.array(self.extractor.extract_features(random_segment, self.sample_rate))

            
            # checa se é nulo
            if not (np.any(extracted_features)):
                continue

            elif( (self.features_df == extracted_features).all(1).any()):
                #print("ELIMINADO POR SER CÓPIA")
                continue
            else:
                # adiciona o segmento aeatório no dataframe final
                self.features_df.loc[len(self.features_df)] = extracted_features
        

        print(f"I generated {len(self.features_df)} samples for training")

        return self.original_segments_only, self.features_df
