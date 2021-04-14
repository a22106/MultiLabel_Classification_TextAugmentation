import pandas as pd
import random 
import transformers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import MultiLabelClassificationModel
import os

import inspect
import re
from sklearn.utils import shuffle

dataset_name = 'HADOOP'

MDL_LABEL_NUM = 89
JRA_LABEL_NUM = 142
ISLANDORA_LABEL_NUM = 67
INFRA_LABEL_NUM = 51
HIVE_LABEL_NUM = 65
HBASE_LABEL_NUM = 68
HADOOP_LABEL_NUM = 37
FCREPO_LABEL_NUM = 22
CONF_LABEL_NUM = 128
CB_LABEL_NUM = 64
CASSANDRA_LABEL_NUM = 15
BAM_LABEL_NUM = 96

class ML_Classification:
    
    labels_Num = {'MDL': MDL_LABEL_NUM, 
    'JRA': JRA_LABEL_NUM, 'ISLANDORA': ISLANDORA_LABEL_NUM, 
    'INFRA': INFRA_LABEL_NUM, 'HIVE': HIVE_LABEL_NUM, 'HBASE': HBASE_LABEL_NUM, 'HADOOP': HADOOP_LABEL_NUM, 'FCREPO': FCREPO_LABEL_NUM, 'CONF': CONF_LABEL_NUM,
     'CB': CB_LABEL_NUM, 'CASSANDRA': CASSANDRA_LABEL_NUM, 'BAM': BAM_LABEL_NUM
     }
    
    def __init__(self, dataset_name = dataset_name, labels_num = labels_Num[dataset_name]):
        self.nlp_model = {'bert': 'bert-base-cased', 'roberta': 'roberta-base', 'xlnet': 'xlnet-base-cased', 'distilbert': 'distilbert-base-cased', 'xlm': 'xlm-roberta-base', 'electra': 'google/electra-base-discriminator'}
        self.nlp_model_name = 'distilbert'
        self.dataset_name = dataset_name
        self.labels_num = labels_num

#       self.data_location = '/home/a22106/python_practice/component_classification/DeepSoft-C/Dataset/Data/{}.csv'.format(self.dataset_name)
        
        # 데이터 위치
        self.data_location_aug = 'HADOOP_aug.csv'
        self.data_location_ori = 'HADOOP.csv'

        # 데이터 변수 입력
        self.data = pd.read_csv(self.data_location_aug) # 증강 데이터
        self.data_ori = pd.read_csv(self.data_location_ori) # 원본 데이터
        self.eval_index = []


    def refine_origin_data(self):
        data = self.data_ori
        
        data_onehot = data.drop(columns = ['issuekey', 'title', 'description', 'component'])
        data_label = []
        for i in range(len(data_onehot)):
            data_label.append(list(data_onehot.iloc[i]))

        # make 'data' value
        data_text = pd.Series(list(data["title"] + ' ' + data['description']), index = data.index)

        #data = data.drop(columns = ['issuekey', 'title', 'description', 'component'])
        self.data_ori = pd.DataFrame(data = {'text': data_text, 'labels': data_label})

        # 오리지날 데이터를 train, eval데이터로 분리
        self.train_data_ori, self.eval_data_ori = train_test_split(self.data_ori, test_size = 0.3)
        self.eval_data = self.eval_data_ori
    

    # 불러온 정제된 데이터 one hot을 str에서 list로 바꾸는 작업
    def labels_to_int(self):
        data = self.data
        changeChar = ' [],'
        for i in range(len(data)):
            for chanChar in changeChar:
                data['labels'][i] = data['labels'][i].replace(chanChar, '')
            data['labels'][i] = list(data['labels'][i])
            data['labels'][i] = list(map(int, data['labels'][i]))
        
        # 증강데이터의 train_data에서 evaluation부분 제거
        
        eval_index_list = list(self.eval_data.index)
        
        for aug_num in range(4):
            iidf2 = [i + 6152* aug_num for i in eval_index_list]
            self.eval_index = self.eval_index + iidf2
        data = self.data

# train, evaluation 데이터로 나누기 // 사용 안함
    def split_data_eval(self, random_state = None):
        data = self.data
        train_data_ori, eval_data_ori = train_test_split(self.data_ori, test_size = 0.3, random_state = random_state)

        eval_data_indexList = eval_data_ori.index.append(eval_data_ori.index * 2)
        data = data.drop(eval_data_indexList)
        
        self.train_data, self.eval_data = train_test_split(data, test_size = 0.3, random_state = random_state)


# 모델 parameter 설정
    def set_model(self):
        self.model = MultiLabelClassificationModel(self.nlp_model_name, self.nlp_model[self.nlp_model_name], num_labels = self.labels_num, 
        args = {'output_dir': '/data/a22106/Classify_/Outputs/HADOOP/{}/outputs20/'.format(self.nlp_model_name), 'overwrite_output_dir': False, 'num_train_epochs':400, 'gradient_accumulation_steps': 2, 'max_seq_length': 128, 'learning_rate': 5e-5})

    def train_model(self):
        self.model.train_model(self.train_data)
    
    def eval_model(self):
        self.result, self.model_outputs, self.wrong_predictions = self.model.eval_model(self.eval_data)
    
    #def predict(self):
    #    self.preds, self.outputs = self.model.predict(self.test_data)

ml = ML_Classification()

ml.refine_origin_data()
ml.labels_to_int()
ml.set_model()
#ml.train_model()
#ml.eval_model()