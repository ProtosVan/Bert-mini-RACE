import sys
import os


root = os.path.dirname(os.path.abspath(__file__))

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import json
import yaml
import logging
import argparse
import random
from tqdm import tqdm
import socket

import torch
from torch.utils import data
from transformers import BertTokenizer, BertModel, BertConfig

from tools.read_dataset import read_dataset, DatasetGenerator
from tools.arg_parser_eval import get_parser
from models.BERT import BertForRace

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    
    if args.config is not None:
        with open(args.config, 'r') as temp_file:
            default_args = yaml.load(temp_file, Loader=yaml.FullLoader)
        keys = vars(args).keys()
        for key in default_args.keys():
            if key not in keys:
                print('Wrong arg: {}'.format(key))
                assert (key in keys)
        parser.set_defaults(**default_args)
        args = parser.parse_args()

    try:
        os.makedirs(args.logs_path)
    except OSError:
        input("Path already exist, check your config.\n(or press ENTER to continue)")
        pass
    
    logging.basicConfig(format = '%(asctime)s: %(levelname)s [%(name)s: %(lineno)d] %(message)s',
                        level = logging.INFO,
                        filename=os.path.join(args.logs_path, 'train.log'),
                        filemode='a')

    logger = logging.getLogger(__file__)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        logger.error("No GPU. Exiting...")
        quit()
    
    logger.info("GPU nums:"+str(torch.cuda.device_count()))
    print("GPU nums:"+str(torch.cuda.device_count()))

    print("Loading model...")
    logger.info("Loading model...")
    checkpoint1 = torch.load(args.model1_path)
    checkpoint2 = torch.load(args.model2_path)
    checkpoint3 = torch.load(args.model3_path)
    
    pretrained_name = args.pre_trained_model_name
    tokenizer = BertTokenizer.from_pretrained(pretrained_name)
    model = BertModel.from_pretrained(pretrained_name)
    config = BertConfig.from_pretrained(pretrained_name)
    
    bertForRace1 = BertForRace(model, config, 0)
    bertForRace1.load_state_dict(checkpoint1['model_state_dict'])
    
    bertForRace2 = BertForRace(model, config, 0)
    bertForRace2.load_state_dict(checkpoint2['model_state_dict'])
    
    bertForRace3 = BertForRace(model, config, 0)
    bertForRace3.load_state_dict(checkpoint3['model_state_dict'])
    
    bertForRace3 = bertForRace3.to(device)
    bertForRace2 = bertForRace2.to(device)
    bertForRace1 = bertForRace1.to(device)
    
    bertForRace1.eval()
    bertForRace2.eval()
    bertForRace3.eval()

    print("Loading model done.")
    logger.info("Loading model done.")

    print("Creating server...")
    logger.info("Creating server...")
    
    serv_socket = socket.socket()
    serv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serv_socket.bind(("127.0.0.1", 21209))
    
    print("Listening...")
    logger.info("Listening...")
    serv_socket.listen()
    
    
    while True:
        serv_connection, serv_addr = serv_socket.accept()
        print("Connection accepted from %s (port:%d)."%(serv_addr[0], serv_addr[1]))
        logger.info("Connection accepted from %s (port:%d)."%(serv_addr[0], serv_addr[1]))

        client_return = serv_connection.recv(10240)
        temp_json = json.loads(client_return.decode('utf-8'))
        if temp_json.has_key("article") and temp_json.has_key("question") and temp_json.has_key("options"):
            pass
        else:
            serv_connection.send(bytes("Wrong input.", encoding='utf-8'))
            serv_connection.close()
            print("Connection closed.")
            logger.info("Connection closed.")
            continue
        
        temp_article = temp_json["article"]
        temp_question = temp_json["question"]
        temp_options = temp_json["options"]
        
        temp_input = []
        for option in temp_options:
            temp_string = temp_article+" [SEP] "+temp_question+" "+option
            temp_token_ids = tokenizer.encode(temp_string, max_length = 512, padding = 'max_length')
            temp_input.append(temp_token_ids)
            
        temp_input = np.array([temp_input]).astype(np.float32)
        temp_input = torch.tensor(temp_input).to(device)
        out1 = bertForRace1(temp_input.long())
        out2 = bertForRace2(temp_input.long())
        out3 = bertForRace3(temp_input.long())
        out_np1 = out1[0].detach().cpu().numpy().tolist()
        out_np2 = out2[0].detach().cpu().numpy().tolist()
        out_np3 = out3[0].detach().cpu().numpy().tolist()
        
        info={"model1":[out_np1[0],out_np1[1],out_np1[2],out_np1[3]]}
        json_info = json.dumps(info)
        serv_connection.send(bytes(json_info, encoding='utf-8'))
        
        
        serv_connection.close()
        print("Connection closed.")
        logger.info("Connection closed.")
