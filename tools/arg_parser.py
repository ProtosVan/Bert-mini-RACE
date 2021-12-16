import os
import sys
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--pre_trained_model_name', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--logs_path', type=str)
    parser.add_argument('--resume_training', type=bool)
    parser.add_argument('--resume_path', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--lock_para', type=bool)
    parser.add_argument('--dropout_rate', type=int)
    return parser