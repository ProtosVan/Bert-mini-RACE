import sys
import os
import torch
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
from torch.utils import data
import json
import shutil
import argparse

def answer_to_number(str):
    if str == "A":
        return 0
    if str == "B":
        return 1
    if str == "C":
        return 2
    if str == "D":
        return 3

def encode_index(strin, input1):
    if strin[0] == 'h':
        # print(strin)
        strin = strin[:-4]
        return_str = "999"+strin[4:]+str(input1)
        # print(return_str)
        # quit()
    else:
        # print(strin)
        strin = strin[:-4]
        return_str = "666"+strin[6:]+str(input1)
        # print(return_str)
        # quit()
    return int(return_str)

def generate_RACE_dataset(old_dataset_path, new_dataset_path, tokenizer, max_length=512):
    os.mkdir(new_dataset_path)

    save_index = []
    save_dataset = []
    save_labels = []
    
    delete_num = 0
    
    for file_path, _, file_names in os.walk(old_dataset_path):
        print(len(file_names))
        for file_name in file_names:
            with open(os.path.join(file_path, file_name), "r") as file_temp:
                json_temp = json.load(file_temp)
            temp_answers = json_temp["answers"]
            temp_options = json_temp["options"]
            temp_questions = json_temp["questions"]
            temp_article = json_temp["article"]
            temp_id = json_temp["id"]
            for temp_i in range(len(temp_questions)):
                # For every question in this article:
                abort_code = False
                temp_datablock=[]
                temp_sequence = temp_article + " [SEP] " + temp_questions[temp_i]
                for temp_j in range(len(temp_options[temp_i])):
                    # For every answer in this question:
                    token_ids = tokenizer.encode(temp_sequence + " " + temp_options[temp_i][temp_j], max_length = max_length, padding = 'max_length')
                    if len(token_ids) > 512:
                        delete_num += 1
                        abort_code = True
                        break
                    temp_datablock.append(token_ids)
                if not abort_code:
                    # Now, datablock has 4 options
                    save_dataset.append(temp_datablock)
                    # 4 options
                    save_labels.append(answer_to_number(temp_answers[temp_i]))
                    # 1 label
                    save_index.append(encode_index(temp_id, temp_i))
                    # 1 id
    print("Delete_num:", delete_num)
    print("Saved index:", len(save_index))
    print("Saved dataset:", len(save_dataset))
    print("Saved labels:", len(save_labels))
    with open(new_dataset_path+"/data.json", "w") as file_temp:
        json.dump([save_index, save_dataset, save_labels], file_temp)
    return
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model",
                        default="bert-base-cased", 
                        type=str,
                        required=True,
                        help="The bert model that you are going to use. This change is along with dataset path.")
    parser.add_argument("--dataset_save_path",
                        default="../../dataset/RACE_bert-base-cased",
                        type=str,
                        required=True,
                        help="The dataset that you are going to use. This change is along with model")
    args = parser.parse_args()
    
    pretrained_name = args.bert_model
    pre_path = args.dataset_save_path
    
    tokenizer = BertTokenizer.from_pretrained(pretrained_name)
    
    if os.path.exists(pre_path):
        shutil.rmtree(pre_path)
    os.mkdir(pre_path)
    
    generate_RACE_dataset("../../dataset/RACE/train/high", pre_path+"/train_h", tokenizer)
    generate_RACE_dataset("../../dataset/RACE/train/middle", pre_path+"/train_m", tokenizer)
    generate_RACE_dataset("../../dataset/RACE/dev/high", pre_path+"/dev_h", tokenizer)
    generate_RACE_dataset("../../dataset/RACE/dev/middle", pre_path+"/dev_m", tokenizer)
    generate_RACE_dataset("../../dataset/RACE/test/high", pre_path+"/test_h", tokenizer)
    generate_RACE_dataset("../../dataset/RACE/test/middle", pre_path+"/test_m", tokenizer)
    
    # 18728
    # 6409
    # 1021
    # 368
    # 1045
    # 362