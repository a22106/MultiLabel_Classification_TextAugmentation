import nlpaug.augmenter.word as naw
import csv
import pandas as pd
aug_backTranslation = naw.BackTranslationAug()

issueReport = pd.read_csv("./Chrome_report_Deeptriage.csv")

def augment_data(dataset):
    aug_range = len(dataset)
    f = open('./Chrome_aug.csv', 'w', newline='', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(dataset)
    for x in range(len(dataset)):
        wr.writerow([dataset['id'][x], dataset['issue_id'][x], dataset['issue_title'][x], 
            dataset['report_time'][x], dataset['report_time'][x], dataset['owner'][x], 
            dataset['description'][x]])
    
    
    for line in range(len(dataset)):
        augmented_Embeded = aug_backTranslation.augment(dataset['description'][line])
        wr.writerow([dataset['id'][line], dataset['issue_id'][line], dataset['issue_title'][line],
            dataset['report_time'][line], dataset['report_time'][line], dataset['owner'][line], augmented_Embeded])
    
    f.close()

augment_data(issueReport)
