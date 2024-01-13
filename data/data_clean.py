import os
import queue 
from bs4 import BeautifulSoup 
from tools.has_punctuation import has_punctuation
import json

root_path = 'data\original_text'
path_queue = queue.Queue()
path_queue.put(root_path)
count = 0
sentences = []

def remove_html_tags(text):  
    soup = BeautifulSoup(text, 'html.parser')  
    return soup.get_text()  

while not path_queue.empty():
    path = path_queue.get()
    file_list = os.listdir(path)
    for file in file_list:
        if file == 'text.txt':
            # print(path)
            with open(os.path.join(path,file), 'r', encoding='utf-8') as txt:
                for line in txt:
                    clean_text = remove_html_tags(line)
                    if has_punctuation(clean_text) and len(clean_text)>30:
                        sentences.append(clean_text)
                    count= count+1
            print(path)
        else:
            new_path = os.path.join(path,file)
            path_queue.put(new_path)

with open('data/original_sentences.json', 'w',encoding='utf-8') as file:  
    json.dump(sentences, file,ensure_ascii=False)

with open('data/original_sentences.txt', 'w',encoding='utf-8') as file:  
    for item in sentences:  
        file.write("%s" % item)
print(count)


