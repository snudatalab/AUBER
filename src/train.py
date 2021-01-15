"""
AUBER: Automated BERT-Regularization

Authors:
- Hyun Dong Lee (hl2787@columbia.edu)
- Seongmin Lee (ligi214@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab at Seoul National University.

File: src/train.py
- Contains source code for running AUBER.
"""

import torch
from transformers import *
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
from scipy.special import softmax

import math
import random
import numpy as np
from itertools import count
import time
from copy import deepcopy
import os
import shutil

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils.default_param import get_default_param
from utils.utils import to_tuple, split_train
from lib.agent import DQN, select_action, optimize_model, get_state
from lib.memory import ReplayMemory
from lib.reward import get_reward

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer, 12)
}

TARGET_UPDATE = 35

def train(model_type,task,original_dir,opt,num_episodes,eval_script,train_script,gpu_num,resume,lr,split,state_m):
    '''
    prune the model
    :param model_type: model type (e.g. bert)
    :param task: GLUE task (e.g. MRPC, MNLI)
    :param original_dir: directory of original finetuned model
    :param opt: optimizer type
    :param num_episodes: number of episodes
    :param eval_script: evaluation script
    :param train_script: train script
    :param gpu_num: gpu for running the script
    :param resume: resume training
    :param lr: learning rate for finetuning BERT
    :param split: split training data set
    '''

    config_class, model_class, tokenizer_class, num_layers = MODEL_CLASSES[model_type]

    # Load original model to prune
    original_model = model_class.from_pretrained(original_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output directory for pruned models
    output_dir = './trained_models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if resume:
        starting_layer = int(original_dir.split('_')[-5]) + 1
    else:
        starting_layer = 0

    # iterate over each layer of model
    for layer in range(num_layers):

        if resume and layer < starting_layer:
            continue

        # output directory for pruning this layer
        layer_output_dir = output_dir + '/{}_{}_{}_{}_{}'.format(model_type,task,state_m,str(layer),lr)
        if not os.path.exists(layer_output_dir):
            os.makedirs(layer_output_dir)
        src_files = os.listdir(original_dir)
        for file_name in src_files:
            full_file_name = os.path.join(original_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, layer_output_dir)

        # initialize policy net, target net, and optimizer
        policy_net = DQN().to(device)
        target_net = DQN().to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.SGD(policy_net.parameters(), lr=0.03, momentum=0.9) 

        if split:
            split_train('../data/{}/train/train.tsv'.format(task))

        # get initial rewards on dev/train datasets
        print('#####################################################')
        dev_reward = get_reward(original_model,layer_output_dir,'dev',False,eval_script,train_script,gpu_num,lr)
        print('original dev accuracy: ', dev_reward)
        print('#####################################################')
        original_reward = get_reward(original_model,layer_output_dir,'train',False,eval_script,train_script,gpu_num,lr)
        print('original train accuracy: ', original_reward)
        print('#####################################################')

        transitions = dict()
        loss_list = []
        episode_reward = []

        memory = ReplayMemory(5000)

        steps_done = 0

        for e in range(num_episodes):
            epi_start_time = time.time()
            model = deepcopy(original_model)
            model.eval()

            prune_list = []
            
            state = torch.Tensor(get_state(model, layer, state_m)).to(device)
            done = False
            prev_reward = original_reward
            current_reward = original_reward
            print('previous reward', prev_reward)
            evaluate = False
            episode_pruned_num = 0

            # repeat until action == 12
            for t in count():
                if episode_pruned_num != 11: # prune specific attention head
                    action = torch.tensor([[select_action(policy_net,state,evaluate,steps_done,prune_list)]], device = device)
                else: # quit pruning
                    action = torch.tensor([[12]], device=device, dtype=torch.long)

                to_prune = action.item()

                steps_done += 1

                print('action: ', action)

                if to_prune != 12:
                    # print('prune')
                    prune = {layer: [to_prune]}
                    model.prune_heads(prune)
                    episode_pruned_num += 1
                elif to_prune == 12:
                    # print('not prune')
                    pass

                if to_prune == 12:
                    done = True
                    next_state = None
                    reward = torch.Tensor([0])
                    # print("final reward: ", current_reward)
                elif to_prune != 12:
                    next_state = deepcopy(state)
                    next_state[0,to_prune] = 0.
                    
                    ns = to_tuple(next_state)
                    
                    # if reward for a state is found, use it
                    if ns in transitions.keys():
                        # print('transition found')
                        current_reward = transitions[ns]
                    else: # otherwise, it is a new state, so evaluate it
                        # print('new transition')
                        current_reward = get_reward(model,layer_output_dir,'train',False,eval_script,train_script,gpu_num,lr)
                        transitions[ns] = current_reward
                    prune_list.append(to_prune)
                    done = False
                        
                    # print('current reward: ', current_reward)
                    reward = (current_reward - prev_reward) * 100
                    
                # store current state/action/next_state/reward info in memory
                memory.push(state, action, next_state, reward, done)

                print('reward: ', reward)
                
                state = deepcopy(next_state)
                prev_reward = current_reward
                print('-----------------------------------')
                
                # Perform one step of the optimization (on the target network)
                if done:
                    print('memory length: ',len(memory))
                    loss = optimize_model(memory, policy_net, target_net, optimizer)
                    loss_list.append(loss)
                    print('loss: ', loss)
                    episode_reward.append(current_reward)
                    print('#############END OF LAYER {} EPISODE {}###############'.format(str(layer),str(e+1)))
                    break

            # Update the target network, copying all weights and biases in DQN
            if e % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            epi_duration = time.time() - epi_start_time
            print("For layer {}, Epi {}, it took {} sec".format(layer, e, epi_duration))

        # finetune the final model and evaluate it on the dev dataset
        print('################VALIDATING##################')
        val_model = deepcopy(original_model)
        val_model.eval()
        val_prune = []
        state = torch.Tensor(get_state(val_model,layer,state_m)).to(device)
        evaluate = True
        while True:
            if len(val_prune) != 11:
                action = torch.tensor([[select_action(policy_net,state,evaluate,1,val_prune)]], device = device)
            else:
                action = torch.tensor([[12]], device=device, dtype=torch.long)
                
            to_prune = action.item()

            if to_prune != 12:
                print('prune ', to_prune)   
                prune = {layer: [to_prune]}
                val_model.prune_heads(prune)
                val_prune.append(to_prune)
                val_reward = get_reward(val_model, layer_output_dir, 'dev', False, eval_script, train_script, gpu_num, lr)
                print("***** PRUNING *****", [layer, to_prune, val_reward.item()])
            elif to_prune == 12:
                if len(val_prune) != 0:
                    val_reward = get_reward(val_model,layer_output_dir,'dev',True,eval_script,train_script,gpu_num,lr)
                elif len(val_prune) == 0:
                    val_reward = get_reward(val_model,layer_output_dir,'dev',False,eval_script,train_script,gpu_num,lr)
                break
            
            state[0,to_prune] = 0.
        print('Validation reward: ', val_reward.item())
        print('#####################################################')

        # The current model becomes the next original model
        original_model = model_class.from_pretrained(layer_output_dir)

