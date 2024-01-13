import torch
from transformers import BertModel

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
pretrained = BertModel.from_pretrained('pretrained_model/bert-base-chinese')
pretrained = pretrained.to(device)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(768, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 32)
        self.fc5 = torch.nn.Linear(32, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        # out = self.conv1(out.last_hidden_state)
        out = self.fc1(out.last_hidden_state)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = torch.sigmoid(out)
        return out