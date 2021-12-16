import os
import sys
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--model1_path', type=str)
    parser.add_argument('--model2_path', type=str)
    parser.add_argument('--model3_path', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--logs_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--pre_trained_model_name', type=str)
    return parser