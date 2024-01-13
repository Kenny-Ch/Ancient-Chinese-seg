import torch
from datasets import load_dataset
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW
import numpy as np
import os
from bert_fc import Model
from tools import strnum2list

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        # self.dataset = load_dataset(path='lansinuote/ChnSentiCorp', split=split)
        with open('data/dataset/text.txt', 'r', encoding='utf-8') as file:
            self.text = file.readlines()
        with open('data/dataset/lable.txt', 'r', encoding='utf-8') as file:
            self.lable = file.readlines() 
        

    def __len__(self):
        return len(self.lable)

    def __getitem__(self, i):
        text = self.text[i]
        label = self.lable[i]

        return text, label

dataset = Dataset('train')    
token = BertTokenizer.from_pretrained('bert-base-chinese')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 定义设备
os.makedirs("weights", exist_ok=True)
temp_loss = 10000.0
batch_size = 64

def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    max_length=500
    #编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
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
    #对label进行编码
    array=[]
    for text in labels:
        text_array = strnum2list(text,max_length)
        array.append(text_array)
    array = np.array(array)
    labels = torch.tensor(array)


    return input_ids, attention_mask, token_type_ids, labels

#数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

#加载预训练模型
# pretrained = BertModel.from_pretrained('bert-base-chinese').to(device)

#不训练,不需要计算梯度
# for param in pretrained.parameters():
#     param.requires_grad_(True)

#定义下游任务模型



model = Model().to(device) 
# print(model(input_ids=input_ids,
#       attention_mask=attention_mask,
#       token_type_ids=token_type_ids).shape)

#训练
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.BCELoss().to(device) 


model.train()
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    
    input_ids=input_ids.to(device)
    attention_mask=attention_mask.to(device)
    token_type_ids=token_type_ids.to(device)
    labels=labels.to(device)

    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
    out = out.reshape(batch_size,-1)

    labels = labels.to(out.dtype)
    mask = labels == 1 
    filtered_out = out[mask]
    filtered_labels = labels[mask]

    mask2 = out > 0.5
    filtered_out2 = out[mask2]
    filtered_labels2 = labels[mask2]

    loss1 = criterion(filtered_out, filtered_labels)
    loss2 = criterion(filtered_out2, filtered_labels2)
    loss = loss1 * 0.1 + loss2 * 0.9
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    train_loss = loss.item()
    if train_loss <= temp_loss:
        temp_loss = train_loss
        torch.save(model.state_dict(),"weights/test-fc.pth".format(temp_loss))

    if i % 5 == 0:
        # out=out.view(-1)
        filtered_out =  (filtered_out > 0.5).float()
        filtered_out2 =  (filtered_out2 > 0.5).float()
        # labels=labels.view(-1)
        accuracy1 = filtered_out.sum().item()/ filtered_labels.sum().item()
        accuracy2 = filtered_labels2.sum().item()/(filtered_out2.sum().item()+1)
        print(i, loss.item(), accuracy1,accuracy2)

    if i == 10000:
        break