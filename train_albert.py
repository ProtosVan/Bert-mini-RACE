import sys
import os
root = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import json
import yaml
import logging
import argparse
import random
from tqdm import tqdm

import torch
from torch.utils import data
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig

from tools.read_dataset import read_dataset, DatasetGenerator
from tools.arg_parser import get_parser
from models.ALBERT import AlbertForRace

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

    if args.resume_training :
        print("Resume training...")
        checkpoint = torch.load(args.resume_path+'/model.pth')
        
        pretrained_name = args.pre_trained_model_name
        tokenizer = AlbertTokenizer.from_pretrained(pretrained_name)
        model = AlbertModel.from_pretrained(pretrained_name)
        config = AlbertConfig.from_pretrained(pretrained_name)
        bertForRace = AlbertForRace(model, config)
        bertForRace.load_state_dict(checkpoint['model_state_dict'])
        if args.lock_para:
            optimizer=torch.optim.Adam([bertForRace.fc1.weight, bertForRace.fc1.bias, bertForRace.fc2.weight, bertForRace.fc2.bias], lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer=torch.optim.Adam(bertForRace.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        resume_epoch = checkpoint['epoch']
        max_accu = checkpoint['max_accu']
        logging.basicConfig(format = '%(asctime)s: %(levelname)s [%(name)s: %(lineno)d] %(message)s',
                    level = logging.INFO,
                    filename=os.path.join(args.logs_path, 'train.log'),
                    filemode='a')
    else:
        print("Not resume training, loading pre-trained model.")
        pretrained_name = args.pre_trained_model_name
        tokenizer = AlbertTokenizer.from_pretrained(pretrained_name)
        model = AlbertModel.from_pretrained(pretrained_name)
        config = AlbertConfig.from_pretrained(pretrained_name)
        bertForRace = AlbertForRace(model, config)
        if args.lock_para:
            optimizer=torch.optim.SGD([bertForRace.fc1.weight, bertForRace.fc1.bias, bertForRace.fc2.weight, bertForRace.fc2.bias], lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer=torch.optim.SGD(bertForRace.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        resume_epoch = 0
        max_accu = -1
        logging.basicConfig(format = '%(asctime)s: %(levelname)s [%(name)s: %(lineno)d] %(message)s',
                    level = logging.INFO,
                    filename=os.path.join(args.logs_path, 'train.log'),
                    filemode='w')

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
    
    train_h_dataset = read_dataset(args.dataset_path + "/train_h/data.json")
    train_dataloader = data.DataLoader(train_h_dataset, batch_size=batch_size)
    
    dev_h_dataset = read_dataset(args.dataset_path + "/dev_h/data.json")
    dev_dataloader = data.DataLoader(dev_h_dataset, batch_size=batch_size)

    bertForRace = bertForRace.to(device)
    bertForRace = torch.nn.DataParallel(bertForRace)

    loss_func = torch.nn.CrossEntropyLoss()
    
    logger.info("Train dataset size:%d"%len(train_h_dataset))
    logger.info("Dev dataset size:%d"%len(dev_h_dataset))
    print("Train dataset size:%d"%len(train_h_dataset))
    print("Dev dataset size:%d"%len(dev_h_dataset))
    logger.info("Learning rate:%f"%args.learning_rate)
    logger.info("Weight decay:%f"%args.learning_rate)
    print("Learning rate:%f"%args.learning_rate)
    print("Weight decay:%f"%args.learning_rate)
    
    
    # Start training    
    print("Begin training...")
    arg_dict = vars(args)
    with open(args.logs_path + '/config.yaml', 'w') as temp_file:
        yaml.dump(arg_dict, temp_file)
    for epoch in range(resume_epoch, args.max_epoch):
        print("Epoch %d start:"%epoch)
        loss_sum=0.0
        accu = 0
        bertForRace.train()
        with tqdm(total=len(train_dataloader), desc='Train', leave=True, unit='batch', unit_scale=True) as pbar:
            for step, (index, tokens, labels) in enumerate(train_dataloader):
                tokens = tokens.to(device)
                labels = labels.to(device)
                out = bertForRace(tokens.long())
                # print(out.shape)
                # print(labels.shape)
                # print(out.size(), labels.size())
                loss = loss_func(out, labels.long())
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum+=loss.cpu().data.numpy()
                accu+=accuracy(out.detach().cpu().numpy(), labels.to('cpu').numpy())
                pbar.update(1)
                # print(accuracy(out.detach().cpu().numpy(), labels.to('cpu').numpy()), len(labels))
                
        print("[TRAIN] Epoch %d, train loss:%f, train accu:%f"%(epoch, loss_sum/len(train_dataloader), accu/len(train_h_dataset)))
        logger.info("[TRAIN] Epoch %d, train loss:%f, train accu:%f"%(epoch, loss_sum/len(train_dataloader), accu/len(train_h_dataset)))
        
        test_loss_sum=0.0
        test_accu = 0
        bertForRace.eval()
        with tqdm(total=len(dev_dataloader), desc='Dev', leave=True, unit='batch', unit_scale=True) as pbar:
            for step, (index, tokens, labels) in enumerate(dev_dataloader):
                tokens = tokens.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    out = bertForRace(tokens.long())
                    loss = loss_func(out, labels.long())
                    loss = loss.mean()
                    test_loss_sum+=loss.cpu().data.numpy()
                    test_accu+=accuracy(out.detach().cpu().numpy(), labels.to('cpu').numpy())
                pbar.update(1)
        print("[TEST] Epoch %d, test loss:%f, test accu:%f"%(epoch, test_loss_sum/len(dev_dataloader), test_accu/len(dev_h_dataset)))
        logger.info("[TEST] Epoch %d, test loss:%f, test accu:%f"%(epoch, test_loss_sum/len(dev_dataloader), test_accu/len(dev_h_dataset)))
        
        if test_accu/len(dev_h_dataset) >= max_accu:
            max_accu = test_accu/len(dev_h_dataset)
            logger.info("Best test accu, saving model to%s"%(args.logs_path))
            print("Best test accu, saving model...")
            torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': bertForRace.module.state_dict(),
                        'max_accu': max_accu
                        }, args.logs_path+'/model.pth')
            print("Save done.")