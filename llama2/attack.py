# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama
from typing import List
import numpy as np
import pandas as pd
import pdb
from tqdm.contrib import tzip
import argparse
import os

MODEL_PATH={
    'llama-2-7b':'llama-2-7b/',
    'llama-2-13b':'llama-2-13b/',
    'llama-2-70b':'llama-2-70b/',
    'llama-2-7b-chat':'llama-2-7b-chat/',
    'llama-2-13b-chat':'llama-2-13b-chat/',
    'llama-2-70b-chat':'llama2-70b-chat'
}

def save_the_text(data_str, path):
    df = pd.DataFrame(data_str)
    # 将DataFrame保存为CSV文件
    df.to_csv(path, index=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, )
    parser.add_argument("--save_path", type=str, default='/root/autodl-tmp/evaluation_output')
    parser.add_argument("--model", type=str,  choices=["beavor","alpaca","llama","llama2-7b","llama2-chat-7b","llama2-chat-13b"])
    parser.add_argument("--task", type=str,  choices=["misspelling","swapping","synonym"])
    parser.add_argument("--eval_len", type=int, default=1000)
    parser.add_argument("--attack_degree", type=int)
    args = parser.parse_args()
    return args

def main(args):
    ckpt_dir = MODEL_PATH[args.model]
    tokenizer_path = 'tokenizer.model'
    temperature: float = 0.6
    top_p: float = 0.9
    max_seq_len: int = 128
    max_gen_len: int = 512
    max_batch_size: int = 2
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    path = '/root/autodl-tmp/prompts' + '/' + args.task + '/' + args.dataset + '/' + 'attack_degree' + str(args.attack_degree)
    text_origin = np.array(pd.read_csv(path+'/origin_prompts.csv')).squeeze(-1).tolist()
    text_attack = np.array(pd.read_csv(path+'/attack_prompts.csv')).squeeze(-1).tolist()

    prompts_cleans = []
    prompts_attacks =[]
    answer_cleans = []
    answer_attacks = []

    for i,(clean_prompt) in enumerate(tzip(text_origin)):
        RLHF_answer_clean = generator.text_completion(
            (clean_prompt),
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        prompts_cleans.append(clean_prompt)
        answer_cleans.append(RLHF_answer_clean[0]['generation'])

    for i,(attack_prompt) in enumerate(tzip(text_attack)):
        RLHF_answer_attack = generator.text_completion(
            attack_prompt,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        prompts_attacks.append(attack_prompt)
        answer_attacks.append(RLHF_answer_attack[0]['generation'])

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

if __name__ == "__main__":
    args = get_args()
    main(args)

