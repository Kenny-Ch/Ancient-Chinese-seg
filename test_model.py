import torch
# from datasets import load_dataset
# from transformers import BertTokenizer
# from transformers import BertModel
# from transformers import AdamW
# token = BertTokenizer.from_pretrained('bert-base-chinese')
# pretrained = BertModel.from_pretrained('bert-base-chinese')
import numpy as np


#不训练,不需要计算梯度
# for param in pretrained.parameters():
#     param.requires_grad_(False)

#模型试算
# out = pretrained(input_ids=input_ids,
#            attention_mask=attention_mask,
#            token_type_ids=token_type_ids)


# original_array = ["0100000","00110011000100","11111101111011"]
# array=[]
# for text in original_array:
#     text_array = []
#     if len(text)<50:
#         for c in text:
#             text_array.append(int(c))
#         text_array = text_array + [0]*(50-len(text))
#     else:
#         for i in range(50):
#             text_array.append(text[i])
#     array.append(text_array)
# array = np.array(array)
# print(array)

# 假设您有一个Tensor，其中包含标签和0  
out = torch.tensor([[0, 0, 0, 1, 0],[0, 0, 0, 1, 0]])  
labels = torch.tensor([[0, 1, 0, 1, 0],[0, 1, 0, 1, 0]])  
  
# 创建一个掩码Tensor，其中0的位置为True，其他位置为False  
mask = labels == 1  
  
# 将掩码Tensor转换为布尔类型（True/False）  
mask = mask.bool()  
  
# 现在，您可以使用掩码来过滤掉Tensor中的0  
filtered_out = out[mask]  
  
print(filtered_out)  # 输出：[1]