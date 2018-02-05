#!/usr/bin/python2
'''
@file run_weights.py
    This script is used to run tests
@author Amber Zhang

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from RLAgent import RLAgent
import gym, gym_mupen64plus
from model import DQNModel  
from keras.utils import plot_model


# Testing Parameters
track_env = 'Mario-Kart-Luigi-Raceway-v0'
model_weights = "model_weigths_01_20_18_03.h5"
test_param = {
    'frame_skips' : 4,
    'nb_frames' : 3,
    'nb_tests' : 1
}


def run_weights():

    env = gym.make('Mario-Kart-Luigi-Raceway-v0')

    resolution = (120, 160)

    actions = [[-60,   0, 1, 0, 0],             # left
               [ 60,   0, 1, 0, 0],             # right
               [  0, -80, 0, 1, 0],             # back
               [  0,   0, 1, 0, 0]]           # go straight
               # [  0,   0, 0, 1, 0]]             # brake

    # Load Model and Weights
    model = DQNModel(resolution=resolution, 
                     nb_frames=test_param['nb_frames'], 
                     actions=actions)
    
    model.load_weights(model_weights)

    agent = RLAgent(model, **test_param)

    agent.test(env)


# run the test process
run_weights()



