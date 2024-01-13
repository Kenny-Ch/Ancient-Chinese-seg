import time
import torch
from bert_fc import Model
from transformers import BertTokenizer
from tools import strnum2list

def predict(word):
    t1 = time.time()
    data = token.encode_plus(text=word,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=max_length,
                                   return_tensors='pt',
                                   return_length=True)
    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    model.train()
    input_ids=input_ids.to(device)
    attention_mask=attention_mask.to(device)
    token_type_ids=token_type_ids.to(device)
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
    out = out.view(-1)
    out = (out > 0.5).float()
    # print(out)
    return out

if __name__ == '__main__':

    # 获取输入内容
    # word = input()
    # orin_word = "及至到了开封府，展爷便先见公孙策商议，求包相保奏白玉堂；然后又与王马张赵彼此见了。众人见白玉堂少年英雄，无不羡爱。白玉堂到此时也就循规蹈矩，诸事仗卢大爷提拨。"
    # print(orin_word)
    # word = "及至到了开封府展爷便先见公孙策商议求包相保奏白玉堂然后又与王马张赵彼此见了众人见白玉堂少年英雄无不羡爱白玉堂到此时也就循规蹈矩诸事仗卢大爷提拨"

    # 从数据集中选择指定数量语句
    index = 3552
    text=[]
    labels=[]
    with open('data/dataset/text.txt', 'r', encoding='utf-8') as file:
        text = file.readlines()
    with open('data/dataset/lable.txt', 'r', encoding='utf-8') as file:
        labels = file.readlines()
    word = text[index]
    print("原句：",end='')
    for i,c in enumerate(word):
        if labels[index][i]=='1':
            print(c+' ',end='')
        else:
            print(c,end='')

    # 实例化网络模型
    max_length = 500
    token = BertTokenizer.from_pretrained('pretrained_model/bert-base-chinese')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    weight_path = 'weights/test-fc.pth'
    model = Model().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    result=predict(word)
    count=0
    # print(result)
    print("预测：",end='')
    for i,c in enumerate(word):
        if result[i].any():
            print(c+' ',end='')
        else:
            print(c,end='')
    
    # 计算相关指标
    labels = torch.tensor(strnum2list(labels[index],500)).to(device)
    mask1 = labels == 1
    filtered_out = result[mask1]
    filtered_labels = labels[mask1]
    print("召回率 R:"+str(filtered_out.sum().item()/filtered_labels.sum().item()))
    mask2 = result == 1
    filtered_out2 = result[mask2]
    filtered_labels2 = labels[mask2]
    print("精确率 P:"+str(filtered_labels2.sum().item()/filtered_out2.sum().item()))
