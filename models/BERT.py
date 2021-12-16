import sys
import os
import torch

class BertForRace(torch.nn.Module):
    def __init__(self, bert_model, bert_config, dropout_rate):
        super().__init__()
        self.bert_model = bert_model
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc1 = torch.nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.fc2 = torch.nn.Linear(bert_config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, token_ids):
        # Batch * options * token_length
        batch_size, options_num, token_length= token_ids.size()
        token_ids = token_ids.reshape([batch_size * options_num, token_length])
        bert_out = self.bert_model(token_ids)[1]
        fc1_out = self.fc1(bert_out)
        drop_out = self.dropout(fc1_out)
        fc2_out = self.fc2(drop_out)
        # B, O, 1
        out = fc2_out.view(-1, options_num)
        # B, O
        return out