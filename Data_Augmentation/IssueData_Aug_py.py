import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug

import pandas as pd
import sys
import csv

### Char Augmenter
# Optical character recognition
aug_char_OCR = nac.OcrAug()
# Keyboard
aug_char_Keyboard = nac.KeyboardAug()

### Word Augmenter
# Spelling
aug_word_Spelling = naw.SpellingAug()
#aug_word_WordEmbedng = naw.WordEmbsAug(model_type = 'word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin', action="insert")
#aug_word_TfIdfAug = naw.TfIdfAug(model_path = os.environ.get("MODEL_DIR"), action = 'insert')
aug_word_ContextualWordEmbs = naw.ContextualWordEmbsAug(model_path = 'bert-base-uncased', action = 'insert')
aug_word_Synonym = naw.SynonymAug(aug_src = 'wordnet')
aug_word_Antonym = naw.AntonymAug()
aug_word_Split = naw.SplitAug()
aug_word_BackTranslation = naw.BackTranslationAug(from_model_name = 'transformer.wmt19.en-de', to_model_name = 'transformer.wmt19.de-en')

def refine_data(dataset_name):
    # 원본 HADOOP csv데이터 불러와서 정제하기
    issueReport = pd.read_csv("../Dataset/Deepsoft_IssueData/{}.csv".format(dataset_name))
    issueOnehot = issueReport.drop(columns = ['issuekey', 'title', 'description', 'component'])

    issueLabel = []
    for i in range(len(issueOnehot)):
        issueLabel.append(list(issueOnehot.iloc[i]))

    # make 'data' value
    refined_data = issueReport

    issueText = pd.Series(list(issueReport["title"] + ' ' + issueReport['description']), index = refined_data.index)
    refined_data['text'] = issueText

    refined_data['labels'] = pd.Series(issueLabel, index = refined_data.index)

    refined_data = refined_data.drop(columns = ['issuekey', 'title', 'description', 'component'])
    refined_data = pd.DataFrame(data = {'text':issueText, 'labels': issueLabel})
    return refined_data

def list_data(refined_data):
    # 정제된 데이터 text 리스트화
    data_list = []
    len(refined_data['text'])
    for x in range(len(refined_data['text'])):
        data_list.append(refined_data['text'][x])
    return data_list

#AUG_INTERVAL = 2

def augment_data(dataset_name, dataset, aug_target, augmenter, augmenter_name, augment_times = 6):
    # augmentation 데이터 추가
    if augment_times <= 0:
        return

    aug_range = len(dataset)
    f = open('../Dataset/Deepsoft_IssueData_AUG/{}_{}_{}.csv'.format(dataset_name, aug_target, augmenter_name), 'w', newline='', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(dataset)
    for x in range(len(dataset)):
        wr.writerow([dataset['text'][x], dataset['labels'][x]])
    
    for x in range(augment_times): # aug multip
        for line in range(aug_range):
            augmented_Embeded = augmenter.augment(dataset['text'][line])
            wr.writerow([augmented_Embeded, dataset['labels'][line]])
    
    f.close()


# refine_data, augment_data
dataset_HADOOP = refine_data("HADOOP")
dataset_FCREPO = refine_data("FCREPO")
dataset_ISLANDORA = refine_data("ISLANDORA")