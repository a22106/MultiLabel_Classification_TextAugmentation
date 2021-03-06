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
        self.data_location_aug = 'D:\\OneDrive\\GitHub\\BugTriage_a22106\\a22106\Dataset\\HADOOP_aug_mul3.csv'
        self.data_location_ori = 'D:\\OneDrive\\GitHub\\BugTriage_a22106\\a22106\Dataset\\HADOOP.csv'
        self.aug_mul = 0

        # 데이터 변수 입력
        self.data = pd.read_csv(self.data_location_aug) # 증강 데이터
        self.data_ori = pd.read_csv(self.data_location_ori) # 원본 데이터
        self.eval_index = []


    def refine_origin_data(self):
        data_ori = self.data_ori
        
        data_onehot = data_ori.drop(columns = ['issuekey', 'title', 'description', 'component'])
        data_label = []
        for i in range(len(data_onehot)):
            data_label.append(list(data_onehot.iloc[i]))

        # make 'data' value
        data_text = pd.Series(list(data_ori["title"] + ' ' + data_ori['description']), index = data_ori.index)

        #data = data.drop(columns = ['issuekey', 'title', 'description', 'component'])
        data_ori = pd.DataFrame(data = {'text': data_text, 'labels': data_label})
        self.data_ori = data_ori

        # 오리지날 데이터를 train, eval데이터로 분리
        self.train_data_ori, self.eval_data_ori = train_test_split(data_ori, test_size = 0.3)
        self.eval_data = self.eval_data_ori
        self.train_data = self.train_data_ori
    

    # 불러온 정제된 데이터 one hot을 str에서 list로 바꾸는 작업
    def labels_to_int(self, aug_mul):
        data = self.data[: 6152 * aug_mul] 
        self.aug_mul = aug_mul

        changeChar = ' [],'
        for i in range(len(data)):
            for chanChar in changeChar:
                data['labels'][i] = data['labels'][i].replace(chanChar, '')
            data['labels'][i] = list(data['labels'][i])
            data['labels'][i] = list(map(int, data['labels'][i]))
        
        # 증강데이터의 train_data에서 evaluation부분 제거
        eval_index_list = list(self.eval_data.index)
        
        for aug_num in range(aug_mul):
            iidf2 = [i + 6152* aug_num for i in eval_index_list]
            self.eval_index = self.eval_index + iidf2

        self.train_data = data.drop(self.eval_index)

        data = self.data

# 모델 parameter 설정
    def set_model(self, nlp_model):
        self.nlp_model_name = nlp_model
        self.model = MultiLabelClassificationModel(self.nlp_model_name, self.nlp_model[self.nlp_model_name], num_labels = self.labels_num, 
        args = {'output_dir': '/data/a22106/Classify_/Outputs/HADOOP/{}/outputs_char_keyboard_aug{}/'.format(self.nlp_model_name, self.aug_mul), 
        'overwrite_output_dir': False, 'num_train_epochs':100, 'batch_size': 32, 'max_seq_length': 128, 'learning_rate': 5e-5})

    def train_model(self):
        self.model.train_model(self.train_data)
    
    def eval_model(self):
        self.result, self.model_outputs, self.wrong_predictions = self.model.eval_model(self.eval_data)
    
    #def predict(self):
    #    self.preds, self.outputs = self.model.predict(self.test_data)

# 원본 데이터 multilabel 학습
ml = ML_Classification()
ml.refine_origin_data()
ml.set_model('bert')
ml.train_model()
ml.eval_model()

# 2배 augmentation 데이터 multilabel 학습
ml2 = ML_Classification()
ml2.refine_origin_data()
ml2.labels_to_int(2)
ml2.set_model('bert')
ml2.train_model()
ml2.eval_model()
