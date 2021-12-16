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

import torch
from torch.utils import data
from transformers import BertTokenizer, BertModel, BertConfig

from tools.read_dataset import read_dataset, DatasetGenerator
from tools.arg_parser_eval import get_parser
from models.BERT import BertForRace

def init_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


    print("Loading model...")
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

    init_seed()

    batch_size = args.batch_size
    
    test_dataset = read_dataset([args.dataset_path + "/test_h/data.json", args.dataset_path + "/test_m/data.json"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size)
    
    test_h_dataset = read_dataset([args.dataset_path + "/test_h/data.json"])
    test_h_dataloader = data.DataLoader(test_h_dataset, batch_size=batch_size)
    
    test_m_dataset = read_dataset([args.dataset_path + "/test_m/data.json"])
    test_m_dataloader = data.DataLoader(test_m_dataset, batch_size=batch_size)

    print("Evaluating model1...")
    logger.info("Evaluation model1...")




    bertForRace1 = bertForRace1.to(device)
    bertForRace1.eval()
    bertForRace2 = bertForRace2.to(device)
    bertForRace2.eval()
    bertForRace3 = bertForRace3.to(device)
    bertForRace3.eval()
    
    for temp_dataloader, temp_datasize, temp_name in [[test_m_dataloader, len(test_m_dataset), "model1_test_m"], [test_h_dataloader, len(test_h_dataset), "model1_test_h"]]:
        logger.info("Test dataset size:%d"%temp_datasize)
        print("Test dataset size:%d"%temp_datasize)
        print("Begin evaluating...")
        test_accu = 0

        with tqdm(total=len(temp_dataloader), desc='Model1', leave=True, unit='batch', unit_scale=True) as pbar:
            with open(os.path.join(args.logs_path, temp_name+".json"), "w") as temp_file:
                save_list = []
                for step, (index, tokens, labels) in enumerate(temp_dataloader):
                    tokens = tokens.to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        out = bertForRace1(tokens.long())
                    out_np = out.detach().cpu().numpy()
                    label_np = labels.to('cpu').numpy()
                    for i in range(len(out_np)):
                        save_list.append([out_np[i].tolist(), label_np[i].tolist()])
                    
                    pbar.update(1)
                json.dump(save_list, temp_file)
        print("[TEST] test accu:%f"%(test_accu/temp_datasize))
        logger.info("[TEST] test accu:%f"%(test_accu/temp_datasize))
        
    for temp_dataloader, temp_datasize, temp_name in [[test_m_dataloader, len(test_m_dataset), "model2_test_m"], [test_h_dataloader, len(test_h_dataset), "model2_test_h"]]:
        logger.info("Test dataset size:%d"%temp_datasize)
        print("Test dataset size:%d"%temp_datasize)
        print("Begin evaluating...")
        test_accu = 0

        with tqdm(total=len(temp_dataloader), desc='Model2', leave=True, unit='batch', unit_scale=True) as pbar:
            with open(os.path.join(args.logs_path, temp_name+".json"), "w") as temp_file:
                save_list = []
                for step, (index, tokens, labels) in enumerate(temp_dataloader):
                    tokens = tokens.to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        out = bertForRace2(tokens.long())
                    out_np = out.detach().cpu().numpy()
                    label_np = labels.to('cpu').numpy()
                    for i in range(len(out_np)):
                        save_list.append([out_np[i].tolist(), label_np[i].tolist()])
                    
                    pbar.update(1)
                json.dump(save_list, temp_file)
        print("[TEST] test accu:%f"%(test_accu/temp_datasize))
        logger.info("[TEST] test accu:%f"%(test_accu/temp_datasize))
    for temp_dataloader, temp_datasize, temp_name in [[test_m_dataloader, len(test_m_dataset), "model3_test_m"], [test_h_dataloader, len(test_h_dataset), "model3_test_h"]]:
        logger.info("Test dataset size:%d"%temp_datasize)
        print("Test dataset size:%d"%temp_datasize)
        print("Begin evaluating...")
        test_accu = 0

        with tqdm(total=len(temp_dataloader), desc='Model3', leave=True, unit='batch', unit_scale=True) as pbar:
            with open(os.path.join(args.logs_path, temp_name+".json"), "w") as temp_file:
                save_list = []
                for step, (index, tokens, labels) in enumerate(temp_dataloader):
                    tokens = tokens.to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        out = bertForRace3(tokens.long())
                    out_np = out.detach().cpu().numpy()
                    label_np = labels.to('cpu').numpy()
                    for i in range(len(out_np)):
                        save_list.append([out_np[i].tolist(), label_np[i].tolist()])
                    
                    pbar.update(1)
                json.dump(save_list, temp_file)
        print("[TEST] test accu:%f"%(test_accu/temp_datasize))
        logger.info("[TEST] test accu:%f"%(test_accu/temp_datasize))
