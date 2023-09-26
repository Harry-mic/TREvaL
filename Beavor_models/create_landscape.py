import copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BertConfig, BertTokenizer, BertForMaskedLM
from safe_rlhf.models import AutoModelForScore
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
import pdb
import json
from rich.progress import track
import tqdm
from tqdm.contrib import tzip
import loss_landscapes
import loss_landscapes.metrics
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,4,5,7'
STEPS = 20
PROMPT_SETS ={
    'nq':'',
}
MODEL_PATH={
    'beavor':'/root/autodl-tmp/huggingface_datasets/beaver-7b-v1.0',
    'alpaca':'/home/data_2/why_22/code/safe-rlhf/alpaca-7b-reproduced',
    'llama':'/home/data_2/why_22/code/safe-rlhf/llama-7b-hf'
}
    
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='nq', choices=['nq'])
    parser.add_argument("--mlm_path", type=str, default='/root/autodl-tmp/huggingface_datasets/bert-base-uncased')
    parser.add_argument("--save_path", type=str, default='/root/autodl-tmp/evaluation_output/')
    parser.add_argument("--model", type=str,  choices=["beavor","alpaca","llama"])
    parser.add_argument("--task", type=str, choices=["misspelling","swapping","synonym"])
    parser.add_argument("--eval_len", type=int, default=1000)
    parser.add_argument("--attack_degree", type=int)

    args = parser.parse_args()
    return args

def save_the_text(data_str, path):
    df = pd.DataFrame(data_str)
   
    # 将DataFrame保存为CSV文件
    df.to_csv(path, index=False)

def main(args):
    #load models
    model_path = MODEL_PATH[args.model]
    model_RLHF = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    model_RLHF.train()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    #load datasets
    dataset_path = args.save_path + '/' + args.model + '/' + args.task + '/' + args.dataset + '/' + 'attack_degree' +str(args.attack_degree)
    prompt_cleans = np.array(pd.read_csv(dataset_path+'/csv_clean_prompts.csv'))[:1] #x
    answer_cleans = np.array(pd.read_csv(dataset_path+'/csv_clean_answers.csv'))[:1]#y
    criterion = torch.nn.CrossEntropyLoss()

    #create x and y
    loss = []
    for i,(answer_clean, prompt_clean) in enumerate(tzip(answer_cleans, prompt_cleans)):
        input_ids = tokenizer.encode(str(prompt_clean), return_tensors='pt')#x
        labels = tokenizer.encode(str(prompt_clean)+str(answer_clean), return_tensors='pt') #y
        metric = loss_landscapes.metrics.Loss(criterion, input_ids, labels)
        loss_data_fin = loss_landscapes.random_plane(model_RLHF, metric, 10, STEPS, normalization='filter', deepcopy_model=True)
        loss.append(loss_data_fin)

    csv = np.array(loss).squeeze(0)
    if not os.path.exists('/home/data_2/why_22/code/safe-rlhf/landscape_loss'):
        os.makedirs('/home/data_2/why_22/code/safe-rlhf/landscape_loss')
    save_the_text(csv,'/home/data_2/why_22/code/safe-rlhf/landscape_loss/' + args.model + '.csv')
    
if __name__ == '__main__':
    args = get_args()
    main(args)