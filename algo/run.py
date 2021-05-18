import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from gym import spaces
import copy
import os
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from copy import deepcopy
from gym.envs.registration import registry, register
from pathlib import Path

# tbh i didn't want to figure out package stuff so uh.. this is the hacky way to get around that heh..
#thanks stackoverflow
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import config
from ppo import PPO
from rollout_buffer import RolloutBuffer 
from acmodel import ACModel 
from cartpole import CartpoleGym
from env.environment import TradeEnv

def get_tickers(ticker_file):
    return pd.read_csv(ticker_file)['Tickers']

# Adapted from 6.884 HW4

def run_experiment(acmodel, env, ppo_kwargs, rollout_kwargs = {}, max_episodes=200000, score_threshold=0.8):
    # acmodel_args should be dictionary corresponding to inputs of acmodel
    # ie {num_tickers: 4, time_horizon: 5, etc..}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    acmodel.to(device)

    is_solved = False

    pd_logs, rewards = [], [0]*config.SMOOTH_REWARD_WINDOW
    
    num_frames = 0
    rollouts = RolloutBuffer(acmodel, env, **rollout_kwargs)
    ppo = PPO(acmodel, **ppo_kwargs)

    # # some variables for bookkeeping purposes
    lr = ppo_kwargs['lr']
    checkpoint_path = f'./checkpoints/acmodel_{config.EPISODE_LENGTH}ep_{config.NUM_PAST_STATES}s_rnn{acmodel.recurrent}_{acmodel.num_layers}lay_{acmodel.hidden_size}hid_{lr}lr'

    pbar = tqdm(range(max_episodes))
    for update in pbar:
        rollouts.reset() # resetting the buffer
        total_return, T = rollouts.collect_experience()
        policy_loss, value_loss = ppo.update(rollouts)
        
        num_frames += T
        rewards.append(total_return)
        
        smooth_reward = np.mean(rewards[-config.SMOOTH_REWARD_WINDOW:])

        data = {'episode':update, 'num_frames':num_frames, 'smooth_reward':smooth_reward,
                'reward':total_return, 'policy_loss': policy_loss, 'value_loss': value_loss}

        pd_logs.append(data)

        pbar.set_postfix(data)

        # Early terminate
        if smooth_reward >= score_threshold:
            is_solved = True
            break

        # save results every 500
        if update % 500 == 0:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
            torch.save(acmodel.state_dict(), f'{checkpoint_path}/{update}')

    if is_solved:
        print('Solved!')
    else:
        print('Unsolved. Check your implementation.')
    
    logs = pd.DataFrame(pd_logs).set_index('episode')

    logs.to_pickle(f'{checkpoint_path}/logs')

    return logs

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--num-tickers', type=int)
    parser.add_argument('-th', '--time-horizon', type=int)
    parser.add_argument('-ta', '--num-ta-indicators', type=int)
    parser.add_argument('-r', '--recurrent')
    parser.add_argument('-s', '--hidden-size', type=int)
    parser.add_argument('-l', '--num-layers', type=int)

    parser.add_argument('-lr', '--lr', type=float)
    parser.add_argument('-ti', '--train-iters', type=int)

    parser.add_argument('-ep', '--episode-length', type=int)

    args = parser.parse_args()
    args_namespace = vars(args)

    args_namespace['recurrent'] = args_namespace['recurrent'] == 'True'

    acmodel_kwargs = {k: args_namespace[k] for k in ['num_tickers', 'time_horizon', 'num_ta_indicators', 'recurrent', 'hidden_size', 'num_layers']}
    ppo_kwargs = {k: args_namespace[k] for k in ['lr', 'train_iters']}

    config.EPISODE_LENGTH = args.episode_length
    config.NUM_PAST_STATES = args.time_horizon

    return acmodel_kwargs, ppo_kwargs

if __name__ == '__main__':
    acmodel_kwargs, ppo_kwargs = parse_args(sys.argv[1:])

    acmodel = ACModel(**acmodel_kwargs)

    data = pickle.load(open('train_ta', 'rb'))
    tickers=['KO', 'BWA', 'QCOM', 'PWR', 'HSIC'] # copied from train_ta file

    env = TradeEnv(tickers=tickers, data=data, features=['volume'] + config.TECHNICAL_INDICATORS_LIST, 
                   episode_length=config.EPISODE_LENGTH, num_past_states=config.NUM_PAST_STATES)

    df = run_experiment(acmodel, env, ppo_kwargs, max_episodes=10000, score_threshold=float("inf"))


    # env_name = 'CartpoleWithMemory-v0'
    # if env_name in registry.env_specs:
    #     del registry.env_specs[env_name]
    # register(
    #     id=env_name,
    #     entry_point=f'{__name__}:CartpoleGym',
    # )
    # env = gym.make('CartpoleWithMemory-v0')