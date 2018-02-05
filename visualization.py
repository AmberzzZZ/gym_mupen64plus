#!/usr/bin/python2
'''
@file train.py
	This script is used to visualize the network performance.
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


track_env = 'Mario-Kart-Luigi-Raceway-v0'

model_weights = "model_weigths_01_20_18_03.h5"

learn_param = {
    'learn_algo' : 'dqlearn',
    'exp_policy' : 'e-greedy',
    'frame_skips' : 4,   # 4
    'nb_epoch' : 20,     #100
    'steps' : 120,         # 5000
    'batch_size' : 40,
    'memory_size' : 10000,
    'nb_frames' : 3,
    'alpha' : [.7, 0.7],
    'alpha_rate' : 0.7,
    'alpha_wait' : 10,
    'gamma' : 0.9,
    'epsilon' : [1.0, 1.0],     #[1.0, 0.1]
    'epsilon_rate' : 0.35,
    'epislon_wait' : 10,   #10
    'nb_tests' : 1,     # 20
}

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
else:
	print("Please provide a model_weights file")

agent = RLAgent(model, **learn_param)    

# give a step number randomly to catch a random screen shot
agent.visualize(env)
