#!/usr/bin/python2
'''
@file train.py
	This script is used to train the DQN model.
@author Amber Zhang

'''


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from RLAgent import RLAgent
import gym, gym_mupen64plus
from model import DQNModel    
import keras.backend as K
import numpy as np


# Training Parameters
track_env = 'Mario-Kart-Luigi-Raceway-v0'

model_weights = None
model_weights = "model_weigths_02_02_09_04.h5"
learn_param = {
    'learn_algo' : 'dqlearn',
    'exp_policy' : 'e-greedy',
    'frame_skips' : 4,   # 4
    'nb_epoch' : 200,     #100
    'steps' : 150,         # 5000
    'batch_size' : 40,
    'memory_size' : 10000,
    'nb_frames' : 3,
    'alpha' : [1.0, 0.1],
    'alpha_rate' : 0.7,
    'alpha_wait' : 10,
    'gamma' : 0.9,
    'epsilon' : [.8, 0.4],     #[1.0, 0.1]
    'epsilon_rate' : 0.35,
    'epislon_wait' : 10,   #10
    'nb_tests' : 1,     # 20
}


def train_model():

    # Initiates the env
    env = gym.make('Mario-Kart-Luigi-Raceway-v0')

    resolution = (120, 160)

    actions = [[-60,   0, 1, 0, 0],             # left
    		   [ 60,   0, 1, 0, 0],             # right
    		   [  0, -80, 0, 1, 0],             # back
    		   [  0,   0, 1, 0, 0]]            # go straight
    		   # [  0,   0, 0, 1, 0]]             # brake

    # Initiates Model
    model = DQNModel(resolution=resolution, 
                     nb_frames=learn_param['nb_frames'], 
                     actions=actions)

    # print("number of actions: ", len(doom.actions))   # 16

    if model_weights: 
        model.load_weights(model_weights)
    
    agent = RLAgent(model, **learn_param)    

    # Preform Reinforcement Learning on Scenario
    agent.train(env)



# run the train process
train_model()