import argparse
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertForMaskedLM, BertConfig
# from safe_rlhf.models import AutoModelForScore
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
import pdb
import json
from rich.progress import track
import tqdm
from tqdm.contrib import tzip
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
PROMPT_SETS ={
    'nq':'',
}
MODEL_PATH={
    'beavor':'/root/autodl-tmp/huggingface_datasets/beaver-7b-v1.0',
    'alpaca':'/root/autodl-tmp/huggingface_datasets/alpaca-7b-reproduced',
    'llama':'/root/autodl-tmp/huggingface_datasets/llama-7b-hf'
}
INSERT_LETTER={
    0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',
    6:'g',7:'h',8:'i',9:'j',10:'k',11:'l',
    12:'m',13:'n',14:'o',15:'p',16:'q',17:'r',
    18:'s',19:'t',20:'u',21:'v',22:'w',23:'x',24:'y',25:'z',
}
class Dataset(object):
    def __init__(self):
        self.data = None

    def __len__(self):
        assert self.data is not None, "self.data is None. Please load data first."
        return len(self.data)

    def get_content_by_idx(self, idx, *args):  
        raise NotImplementedError("get_content_by_idx() must be implemented in the subclass.")
    
    def get_few_shot_examples(self, task):
        raise NotImplementedError("get_few_shot_examples() must be implemented in the subclass.")

class NQ(Dataset):
    def __init__(self, dataset_type="validation"):
        self.data = []
        with open("/root/autodl-tmp/selected_prompts.json", "r") as file:
            for line in file.readlines():
                self.raw_data = json.loads(line)
                self.data.append(self.raw_data)

    def get_content_by_idx(self, idx, *args):
        content = self.data[idx]['question']
        return content
             
def create_dataset(dataset_name, *args):
    if dataset_name == 'nq':
        return NQ()
    
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['nq'])##
    parser.add_argument("--save_path", type=str, default='/root/autodl-tmp/evaluation_output')
    parser.add_argument("--model", type=str,  choices=["beavor","alpaca","llama"])##
    parser.add_argument("--mlm_path", type=str, default='/root/autodl-tmp/huggingface_datasets/bert-base-uncased')
    parser.add_argument("--task", type=str, choices=["misspelling","swapping","synonym"])##
    parser.add_argument("--eval_len", type=int, default=1000)
    parser.add_argument("--attack_degree", type=int)##

    args = parser.parse_args()
    return args

def swap_elements(tensor, idx1, idx2):
    temp = tensor[idx1].clone()
    tensor[idx1] = tensor[idx2]
    tensor[idx2] = temp
    return tensor

def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '').lower()
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys
def attack(text, args, tokenizer_mlm, mlm_model):
    task = args.task
    degree = args.attack_degree
    if args.task == "misspelling":
        length = len(text)
        type = torch.randint(0, 3, (1,))
        #insert some letters
        if type == 0: 
            key =  torch.randint(0, length, (int(length/degree),))
            for i in range(len(key)):
                insert_letter = INSERT_LETTER[torch.randint(0, 26, (1,)).item()]
                text = text[:key[i]] + insert_letter + text[key[i]:]
        #delete some letters
        elif type == 1:
            key =  torch.randint(1, length-1, (int(length/degree),))
            for i in range(len(key)):
                text = text[:key[i]] + text[key[i]+1:]
        #substitue some letters
        elif type == 2:
            key =  torch.randint(1, length-1, (int(length/degree),))
            for i in range(len(key)):
                insert_letter = INSERT_LETTER[torch.randint(0, 26, (1,)).item()]
                text = text[:key[i]] + insert_letter + text[key[i]+1:]
        return text
    elif args.task == "swapping":
        words, sub_words, keys= _tokenize(text, tokenizer_mlm)
        sub_words = ['[CLS]'] + sub_words[:512 - 2] + ['[SEP]']
        length = len(words)
        swap_time = int(length/degree)
        for j in range(swap_time):
            key1 =  torch.randint(0, length, (1,)).item()
            key2 =  torch.randint(max(key1 - 5, 0), min(key1 + 5, length), (1,)).item()
            while(key1 == key2):
                key2 = torch.randint(max(key1 - 5, 0), min(key1 + 5, length), (1,)).item()
            words[:] = swap_elements(words[:], key1, key2)
        attack_text = tokenizer_mlm.convert_tokens_to_string(words)
        return attack_text
    elif args.task == "synonym":
        words, sub_words, keys= _tokenize(text, tokenizer_mlm)
        k = int(len(words)/degree)
        assert len(words)==len(keys)
        sub_words = ['[CLS]'] + sub_words[:512 - 2] + ['[SEP]']
        input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
        word_predictions = mlm_model(input_ids_.to('cuda'))[0].squeeze()
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, 10, -1)
        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
        length = len(keys)
        pos =  torch.randint(0, length, (k,))
        select = torch.randint(0, 10, (k,))
        for i in range(k):
            substitue = word_predictions[keys[pos[i]][0]:keys[pos[i]][1]].reshape(-1)
            if len(substitue) == 0:
                break
            substitue = substitue[select[i]].item()
            words[pos[i]] = tokenizer_mlm._convert_id_to_token(substitue)
        attack_text = tokenizer_mlm.convert_tokens_to_string(words)
        return attack_text
    else:
        raise("The task format is not support yet.")


def save_the_text(data_str, path):
    df = pd.DataFrame(data_str)
    df.to_csv(path, index=False)

def main(args):
    #load dataset
    model_path = MODEL_PATH[args.model]
    model_RLHF = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    model_RLHF.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if args.task == "swapping" or "synonym" :
        tokenizer_mlm = BertTokenizer.from_pretrained(args.mlm_path, do_lower_case=True)
        config_atk = BertConfig.from_pretrained(args.mlm_path)
        mlm_model = BertForMaskedLM.from_pretrained(args.mlm_path, config=config_atk)
        mlm_model.to('cuda')
    else:
        tokenizer_mlm = None 
        mlm_model = None  

    data = create_dataset(args.dataset)
    prompt_prior = PROMPT_SETS[args.dataset]
    data_len = len(data)
    if data_len > args.eval_len:
        data_len = args.eval_len

    #load text_origin and text_attack
    text_attack = []
    text_origin = []
    for idx in range(data_len):
        raw_data = data.get_content_by_idx(idx, args.dataset)
        text_origin.append(raw_data) 
        attack_data = attack(raw_data, args, tokenizer_mlm, mlm_model)
        text_attack.append(attack_data)
        
    prompts_path = '/root/autodl-tmp/prompts'+ '/'  + args.task + '/' + args.dataset + '/' + 'attack_degree' +str(args.attack_degree)
    if not os.path.exists(prompts_path):
        os.makedirs(prompts_path)
    save_the_text(text_origin, prompts_path+'/origin_prompts.csv')
    save_the_text(text_attack, prompts_path+'/attack_prompts.csv')

    prompts_cleans = []
    prompts_attacks =[]
    answer_cleans = []
    answer_attacks = []
    prompt = 'BEGINNING OF CONVERSATION: USER: {prompt_prior}{input} ASSISTANT:'

    for i,(attack_prompt, clean_prompt) in enumerate(tzip(text_attack, text_origin)):

        input_attack = prompt.format(prompt_prior = prompt_prior, input = attack_prompt)
        input_clean = prompt.format(prompt_prior = prompt_prior, input= clean_prompt)
        input_ids_attack = tokenizer.encode(input_attack, return_tensors='pt', max_length= 512, truncation=True).cuda()
        input_ids_clean = tokenizer.encode(input_clean, return_tensors='pt', max_length= 512, truncation=True).cuda()

        # output_ids
        output_ids_RLHF_attack = model_RLHF.generate(input_ids_attack, max_new_tokens = 512)[0]
        output_ids_RLHF_clean = model_RLHF.generate(input_ids_clean, max_new_tokens = 512)[0]

        RLHF_answer_attack = tokenizer.decode(output_ids_RLHF_attack)[4:]
        RLHF_answer_clean = tokenizer.decode(output_ids_RLHF_clean)[4:]

        prompts_cleans.append(input_clean)
        prompts_attacks.append(input_attack)
        answer_cleans.append(RLHF_answer_clean[len(input_clean):])
        answer_attacks.append(RLHF_answer_attack[len(input_attack):])

    csv_clean_prompts = np.array(prompts_cleans)
    csv_attack_prompts = np.array(prompts_attacks)
    csv_clean_answers = np.array(answer_cleans)
    csv_attack_answers = np.array(answer_attacks)

    save_path = args.save_path + '/' + args.model + '/' + args.task + '/' + args.dataset + '/' + 'attack_degree' +str(args.attack_degree)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_the_text(csv_clean_prompts, save_path + '/csv_clean_prompts.csv')
    save_the_text(csv_attack_prompts,  save_path + '/csv_attack_prompts.csv')
    save_the_text(csv_clean_answers,  save_path + '/csv_clean_answers.csv')
    save_the_text(csv_attack_answers,  save_path +'/csv_attack_answers.csv')
    
if __name__ == '__main__':
    args = get_args()
    main(args)