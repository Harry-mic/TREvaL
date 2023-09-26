from transformers import AutoTokenizer
from safe_rlhf.models import AutoModelForScore
import pandas as pd
import numpy as np
import pdb
from tqdm.contrib import tzip
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
MODEL_PATH={
    'RewardModel':'/beaver-7b-v1.0-reward',
    'CostModel':'/beaver-7b-v1.0-cost',
}

def save_the_text(data_str, path):
    df = pd.DataFrame(data_str)
   
    # 将DataFrame保存为CSV文件
    df.to_csv(path, index=False)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=["sst2", "cola", "qqp", 
                                                                        "mnli", "mnli_matched", "mnli_mismatched", 
                                                                        "qnli", "wnli", "rte", "mrpc",
                                                                        "mmlu", "squad_v2", "un_multi", "iwslt", "math",'imdb','nq'
                                                                        ])
    parser.add_argument("--save_path", type=str, default='/root/autodl-tmp/evaluation_output')
    parser.add_argument("--model", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--attack_degree", type=int)

    args = parser.parse_args()
    return args

def main(args):
    reward_model = AutoModelForScore.from_pretrained(MODEL_PATH[args.model], device_map='auto')
    reward_model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[args.model], use_fast=False)
    path = args.save_path + '/' + args.model + '/' + args.task + '/' + args.dataset + '/' + 'attack_degree' +str(args.attack_degree)

    prompt_cleans = np.array(pd.read_csv(path+'/csv_clean_prompts.csv'))
    answer_cleans = np.array(pd.read_csv(path+'/csv_clean_answers.csv'))
    answer_attacks = np.array(pd.read_csv(path+'/csv_attack_answers.csv'))

    clean_rewards = []
    attack_rewards = []
    for i,(prompt_clean, answer_clean, answer_attack) in enumerate(tzip(prompt_cleans, answer_cleans, answer_attacks)):
        if answer_attack!=answer_attack:
            answer_attack = " "
        if answer_clean!=answer_clean:
            answer_clean = " "
        input_ids_attack = tokenizer(str(prompt_clean + answer_attack), return_tensors='pt').to('cuda')
        input_ids_clean = tokenizer(str(prompt_clean + answer_clean), return_tensors='pt').to('cuda')

        attack_reward = reward_model(**input_ids_attack)
        attack_rewards.append(attack_reward['end_scores'].item())
        clean_reward = reward_model(**input_ids_clean)
        clean_rewards.append(clean_reward['end_scores'].item())

    csv_rewards_clean = np.array(clean_rewards)
    csv_rewards_attack  = np.array(attack_rewards)
    save_the_text(csv_rewards_clean, path+'/csv_rewards_clean.csv')
    save_the_text(csv_rewards_attack, path+'/csv_rewards_attack.csv')

if __name__ == '__main__':
    args = get_args()
    main(args)