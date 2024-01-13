import os
from tqdm import tqdm 

punctuation = ['。','，','：','！','？','、','；','）','（','》','《','［','］','【','】',')','(']
sentences = []
lable = []

with open('data\original_sentences.txt', 'r', encoding='utf-8') as txt:
    lines = txt.readlines()  
    for line in tqdm(lines,desc="处理进度"):
        sent=line.replace(" ", "").replace("\n", "")
        list = []
        sent2=""
        for c in sent:
            if c not in punctuation:
                list.append('0')
            else:
                if len(list)>0:
                    list.pop()
                    list.append('1')

        for c in sent:
            if c not in punctuation:
                sent2=sent2+c
        
        sentences.append(sent2)
        lable.append(''.join(list))

with open('data/dataset/text.txt', 'w',encoding='utf-8') as file:  
    for item in sentences:  
        file.write("%s\n" % item)

with open('data/dataset/lable.txt', 'w',encoding='utf-8') as file:  
    for item in lable:  
        file.write("%s\n" % item)