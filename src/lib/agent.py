"""
AUBER: Automated BERT-Regularization

Authors:
- Hyun Dong Lee (hl2787@columbia.edu)
- Seongmin Lee (ligi214@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab at Seoul National University.

File: src/lib/agent.py
 - Contains source code for reinforcement learning agent.

Citation for class DQN, select_action, and optimize_model
: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import random
import math

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
from scipy.special import softmax
import torch.nn.functional as F

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS_START = 1.
EPS_END = 0.05
EPS_DECAY = 256.
BATCH_SIZE = 128
GAMMA = 1.

# DQN agent for policy net and target net
class DQN(nn.Module):
    def __init__(self, nb_states=12, nb_actions=13, hidden1=512, hidden2=512, hidden3=512):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, nb_actions)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        dummy = torch.Tensor(np.ones(out.shape[0]).reshape(-1,1)).to(device)
        x_mask = torch.cat((x,dummy),1)
        
        # reward for pruning already-pruned attention head is set to -5
        out[x_mask==0] = -5.
        out[:,12] = 0.2

        return out

def select_action(net, state, evaluate, steps_done, prune_list, nb_actions=13):
    '''
    selects the next action when given the policy network
    :param net: policy network
    :param state: input to the policy network
    :param evaluate: whether to strictly use policy network to pick the next state
    :param steps_done: number of steps done
    :param prune_list: list of pruned attention heads for the current episode
    :param nb_actions: number of possible actions
    '''
    #epsilon greedy
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold or evaluate:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # print('policy')
            return net(state).max(1)[1].view(1, 1)
    else:
        x = torch.tensor([[random.randrange(nb_actions)]], device=device, dtype=torch.long)
        while x.item() in prune_list:
            x = torch.tensor([[random.randrange(nb_actions)]], device=device, dtype=torch.long)
        # print('random')
        return x.item()

def optimize_model(memory, policy_net, target_net, optimizer):
    '''
    update the policy network by comparing it with the target network
    :param memory: replay memory
    :param policy_net: policy network
    :param target_net: target network
    :param optimizer: optimizer
    '''

    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = memory.transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None],dim=0)

    state = []
    for s in batch.state:
        state.append(s)
        
    state_batch = torch.cat(state,dim=0)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # print(loss)
    
    return loss

def get_state(model, layer_num, state_m):
    '''
    get the state vector for layer_num
    :param model: the model (e.g. bert, albert) from which the initial state vector will be calculated
    :param layer_num: layer number for calculated the state vector
    '''

    state = []
    if state_m == 'value':
        v = model.bert.encoder.layer[layer_num].attention.self.value.weight
    elif state_m == 'query':
        v = model.bert.encoder.layer[layer_num].attention.self.query.weight
    elif state_m == 'key':
        v = model.bert.encoder.layer[layer_num].attention.self.key.weight
    for i in range(12):
        v_i = v[64*i:64*(i+1),:].detach().numpy().reshape(-1,1)
        v_i = LA.norm(v_i,None)
        state.append(v_i)

    transformer = StandardScaler()

    state = np.asarray(state).reshape(-1,1)
    state = transformer.fit_transform(state)
    state = state.reshape(1,-1)
    state = softmax(state, axis=1)
    
    return state
