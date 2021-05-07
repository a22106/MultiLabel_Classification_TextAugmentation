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
import tensorflow as tf

import json, re, nltk, string

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

labels_Num = {'MDL': MDL_LABEL_NUM, 
    'JRA': JRA_LABEL_NUM, 'ISLANDORA': ISLANDORA_LABEL_NUM, 
    'INFRA': INFRA_LABEL_NUM, 'HIVE': HIVE_LABEL_NUM, 'HBASE': HBASE_LABEL_NUM, 'HADOOP': HADOOP_LABEL_NUM, 'FCREPO': FCREPO_LABEL_NUM, 'CONF': CONF_LABEL_NUM,
    'CB': CB_LABEL_NUM, 'CASSANDRA': CASSANDRA_LABEL_NUM, 'BAM': BAM_LABEL_NUM
    }

class ML_Classification:
    def __init__(self, dataset_name, augmenter_name, augment_size, nlp_model_name):
        self.dataset_name = dataset_name
        self.labels_num = labels_Num[dataset_name]

        if augmenter_name == 'OCR' or augmenter_name == 'Keyboard':
            self.augmentation_type = 'char'
        else:
            self.augmentation_type = 'word'

        self.augmenter_name = augmenter_name
        self.aug_mul = augment_size
        self.nlp_model_name = nlp_model_name
        self.nlp_model = {'bert': 'bert-base-uncased', 'roberta': 'roberta-base', 'xlnet': 'xlnet-base-uncased', 'distilbert': 'distilbert-base-uncased', 'xlm': 'xlm-roberta-base', 'electra': 'google/electra-base-discriminator'}


        # 데이터 위치 data location
        self.data_location_ori = '../Dataset/Deepsoft_IssueData/{}.csv'.format(self.dataset_name)

        self.data_location_aug = '../Dataset/Deepsoft_IssueData_Aug/{}_{}_{}.csv'.format(self.dataset_name, self.augmentation_type, self.augmenter_name)     
        

        # 데이터 변수 입력
        self.data = pd.read_csv(self.data_location_aug) # 증강 데이터
        self.data_ori = pd.read_csv(self.data_location_ori) # 원본 데이터
        self.len_data = len(self.data_ori)
        self.eval_index = []
        self.test_index = []



    def refine_origin_data(self):
        data_ori = self.data_ori
        
        data_onehot = data_ori.drop(columns = ['issuekey', 'title', 'description', 'component'])
        data_label = []
        for i in range(len(data_onehot)):
            data_label.append(list(data_onehot.iloc[i]))

        # make 'data' value
        data_text = pd.Series(list(data_ori["title"] + ' ' + data_ori['description']), index = data_ori.index)
        #data = data.drop(columns = ['issuekey', 'title', 'description', 'component'])
        

        refined_data = []
        for item in data_text:
            #1. Remove \r 
            current_desc = item.replace('\r', ' ')    
            #2. Remove URLs
            current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_desc)    
            #4. Remove hex code
            current_desc = re.sub(r'(\w+)0x\w+', '', current_desc) 
            #5. Change to lower case
            current_desc = current_desc.lower()   
            #6. Tokenize
            #current_desc_tokens = tokenizer(current_desc, add_special_tokens= True)
            #7. Strip trailing punctuation marks    
            #current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]     
            #8. Join the lists
            #current_data = current_desc_filter
            #current_data = list(filter(None, current_data))
            refined_data.append(current_desc)

        #data_ori = pd.DataFrame(data = {'text': refined_data, 'labels': data_label})
        data_ori['text'] = refined_data
        data_ori['labels'] = data_label
        self.data_ori = data_ori
        # 오리지날 데이터를 train, eval데이터로 분리

        train_size = 0.6
        test_size = 0.2
        eval_size = 0.2
        self.train_data_ori, self.eval_data_ori = train_test_split(data_ori, train_size = train_size)
        self.eval_data_ori, self.test_data_ori = train_test_split(self.eval_data_ori, test_size = 0.5)
        self.eval_data = self.eval_data_ori
        self.train_data = self.train_data_ori
        self.test_data = self.test_data_ori


    
    # 불러온 정제된 데이터 one hot을 str에서 list로 바꾸는 작업
    def labels_to_int(self):
        if self.aug_mul <= 1:
            return        

        data = self.data[: self.len_data * self.aug_mul] 

        refined_data = []
        for item in data['text']:
            #1. Remove \r 
            current_desc = item.replace('\r', ' ')    
            #2. Remove URLs
            current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_desc)    
            #4. Remove hex code
            current_desc = re.sub(r'(\w+)0x\w+', '', current_desc) 
            #5. Change to lower case
            current_desc = current_desc.lower()   
            #6. Tokenize
            #current_desc_tokens = tokenizer(current_desc, add_special_tokens= True)
            #7. Strip trailing punctuation marks    
            #current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]     
            #8. Join the lists
            #current_data = current_desc_filter
            #current_data = list(filter(None, current_data))
            refined_data.append(current_desc)

        data['text'] = refined_data

        changeChar = ' [],'
        for i in range(len(data)):
            for chanChar in changeChar:
                data['labels'][i] = data['labels'][i].replace(chanChar, '')
            data['labels'][i] = list(data['labels'][i])
            data['labels'][i] = list(map(int, data['labels'][i]))
        
        # 증강데이터의 train_data에서 evaluation부분 제거
        eval_index_list = list(self.eval_data.index)
        test_index_list = list(self.test_data.index)
        
        for aug_num in range(self.aug_mul):
            iidf2 = [i + self.len_data* aug_num for i in eval_index_list]
            self.eval_index = self.eval_index + iidf2
            iidf3 = [x + self.len_data* aug_num for x in test_index_list]
            self.test_index = self.test_index + iidf3


        self.data = data
        self.train_data = data.drop(self.eval_index)
        self.train_data = self.train_data.drop(self.test_index)
        self.train_data = self.train_data.sample(frac=1).reset_index(drop=True)

# 모델 parameter 설정
    def set_model(self, repeat): # epochs: 200, batch size: 100, learning rate 0.002
        self.model = MultiLabelClassificationModel(self.nlp_model_name, self.nlp_model[self.nlp_model_name], num_labels = self.labels_num, 
        args = {'output_dir': '/data/a22106/Deepsoft_C_Multilabel/{}_{}_{}_{}_re{}/'.format(self.dataset_name, self.nlp_model_name, self.augmenter_name, self.aug_mul, repeat), 
        'overwrite_output_dir': True, 'save_steps': -1, 'num_train_epochs': 200, 'train_batch_size': 100, 'eval_batch_size': 100, 'max_seq_length': 128, 'learning_rate': 0.0002})
        

    def train_model(self):
        self.model.train_model(self.train_data)
    
    def eval_model(self):
        self.result, self.model_outputs, self.wrong_predictions = self.model.eval_model(self.eval_data)

    def test_model(self):
        self.to_predict = list(self.test_data['text'].apply(lambda x: x.replace('\n', ' ')).tolist())
        self.preds, self.outputs = self.model.predict(self.to_predict)



    #def predict(self):
    #    self.preds, self.outputs = self.model.predict(self.test_data)

# Deepsoft-C recall method
def recall(actual, estimate, startK, stopK, stepK):
    recall_k = []
    for k in range(startK, stopK + 1, stepK):
        if k == 0:
            k = 1
        m = len(actual)
        sum_recall = 0.0
        for j in range(len(actual)):
            y_true = np.argwhere(actual[j])
            y_pred = estimate[j].argsort()[-k:]
            # print y_true
            # print y_pred
            intersect = (y_true == y_pred).sum()
            # print intersect
            # print (len(y_true))
            # print intersect / float(len(y_true))
            sum_recall = sum_recall + (intersect / float(len(y_true)))
            # print sum_recall
        recall_k.append(sum_recall / float(m))
        # print m
        print('Recall@' + str(k) + ': {:.4f}'.format(sum_recall / float(m)))
    return recall_k

'''startK = 1
stepK = 1
stopK = 30

print("Project:" + project)

path_output = 'Other_DL/output/'

recall_k = recall(y_test, y_pred, startK, stopK, stepK)

np.savetxt(path_output + project + "_recall_" + str(startK) + "_" + str(stopK)+ "_v.csv", recall_k, delimiter=",", fmt='%1.4f')
with open(path_output + "performance" + "_recall_" + str(startK) + "_" + str(stopK)+ ".csv", 'a') as myoutput:
myoutput.write(project + "," + ",".join(map(str, recall_k)) + '\n')'''

dataset_name = ['FCREPO']
#augmenter_name = ["OCR", "Keyboard", "Spelling", "ContextualWordEmbs", "Synonym", "Antonym", "Split"]
augmenter_name = ["ContextualWordEmbs", "Synonym", "Antonym", "Split"]
nlp_model = ['bert', 'distilbert', 'robert']

'''ml = ML_Classification("HADOOP", "ContextualWordEmbs", 1, 'distilbert')
ml.refine_origin_data()
print(ml.len_data)
print(ml.data)
print(ml.train_data)
print(ml.eval_data)
print(ml.x_train)'''

for dataset in dataset_name:
    for augmenter in augmenter_name:
        for times in range(1, 8, 2):
            for repeat2 in range(5):
                ml = ML_Classification(dataset, augmenter, times, 'distilbert')
                ml.refine_origin_data()
                print(ml.len_data)
                ml.labels_to_int()
                ml.set_model(repeat2)
                print(ml.train_data)
                print(ml.eval_data.sort_index())
                ml.train_model()
                ml.eval_model()
                ml.test_model()
