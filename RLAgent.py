#!/usr/bin/python2
'''
@file RLAgent.py
    This script is used to intermediate with the env. 
    Including train and test iterations.
@author Amber Zhang

'''


import numpy as np
from model import *
from random import sample
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from tqdm import tqdm 
import os
import datetime
from scipy.misc import imresize
from datetime import datetime
import matplotlib.pyplot as plt

class RLAgent:

	def __init__(self, model, learn_algo = 'dqlearn', exp_policy='e-greedy', nb_tests=100, frame_skips=4, nb_epoch=1000, steps=1000, target_update=100,
		batch_size=50, memory_size=1000, nb_frames=1, alpha = [1.0,0.1], alpha_rate=1.0, alpha_wait=0, gamma=0.9, epsilon=[1., .1], epsilon_rate=1.0,
		epislon_wait=0, state_predictor_watch=0):
		'''
		Method initiates learning parameters for Reinforcement Learner.

		'''
		self.model = model
		self.memory = ReplayMemory(memory_size)
		self.prev_frames = None
		self.nb_tests = nb_tests

		# Learning Parameters
		self.learn_algo = learn_algo
		self.exp_policy = exp_policy
		self.nb_epoch = nb_epoch
		self.steps = steps
		self.batch_size = batch_size
		self.nb_frames = nb_frames
		self.frame_skips = frame_skips

		# Set Gamma
		self.gamma = gamma

		# Set Alpha and linear Alpha decay
		self.alpha, self.final_alpha = alpha
		self.alpha_wait = alpha_wait
		self.delta_alpha = ((alpha[0] - alpha[1]) / (nb_epoch * alpha_rate))

		# Set Epsilon and linear Epsilon decay
		self.epsilon, self.final_epsilon = epsilon
		self.epislon_wait = epislon_wait
		self.delta_epsilon =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))


	def get_state_data(self, env, action):

		# return the processed RGB data of several successive frames

		(obs, reward, end, info) = env.step(action) 
		env.render()
		# print(obs.shape)                    # (480, 640, 3) screen pixels
		# process the data 
		screen_buffer = obs.astype('float32')/255     # (480, 640 ,3)

		# Grey Scaling
		screen_buffer = np.transpose(screen_buffer, (2, 0, 1))           #(3, 480, 640)
		color_weights = [0.21, 0.72, 0.07]
		grey_buffer = np.zeros(screen_buffer.shape[1:])
		for i in range(3):
			grey_buffer += screen_buffer[i,:,:] * color_weights[i]
		# compress
		grey_buffer = imresize(grey_buffer, (120, 160))
		# print(grey_buffer.shape)   (120, 480)

		if self.prev_frames is None:
			self.prev_frames = [grey_buffer] * self.nb_frames
		else:
			self.prev_frames.append(grey_buffer)
			self.prev_frames.pop(0)

		return np.expand_dims(self.prev_frames, 0), reward, end, obs        # (1, 3, 120, 160)


	def train(self, env):

		print("Training:")              
		print("Model:", self.model.__class__.__name__)          # DQN
		print("Algorithm:", self.learn_algo)                    # deep Q-learning
		print("Exploration_Policy:", self.exp_policy)           # epsilon-greedy
		print("Frame Skips:", self.frame_skips)                 # 4
		print("Number of Previous Frames Used:", self.nb_frames)# 3
		print("Batch Size:", self.batch_size)             # 40

		training_data = []
		best_score = None
		
		for epoch in range(self.nb_epoch):  

			# training loop
			self.prev_frames = None
			a_prime = 0
			env.reset()
			env.render()
			for i in range(88):
			    S, r, is_game_end, _ = self.get_state_data(env, [0, 0, 0, 0, 0])    # NOOP until green light
			    env.render()
			# # skip the straight piece of track, skip the first 2 checkpoints
			# for i in range(160):
			# 	S, r, is_game_end, _ = self.get_state_data(env, [0, 0, 1, 0, 0])
			# 	env.render()
			# pbar = tqdm(total=self.steps)

			# if epoch:
				# training_data.append([loss, np.mean(rewards), np.max(rewards), np.min(rewards), np.std(rewards)])
			
			step = 0
			loss = 0
			total_reward = 0
			rewards = []	

			while step < self.steps:
				# print("episode: %d, step: %d, frames: %d" % ((epoch+1), (step+1), ((step+1)*(1+self.frame_skips))))
				# Exploration Policies
				if self.exp_policy == 'e-greedy':
					if np.random.random() < self.epsilon:         # explore
						# print("explore")
						q = int(np.random.randint(self.model.nb_actions))   # index
						a = self.model.predict(q)                 # action
					else:                                         # exploit: a = argmaxQ(s,a)
						q = self.model.online_network.predict(S)  # y0: output of the network using current weights, nb_action-dims
						# first q.shape: (1, 16)
						q = int(np.argmax(q[0]))
						a = self.model.predict(q)

				# Advance Action over frame_skips + 1
				if not is_game_end:
					max_r = -1
					for i in range(self.frame_skips+1):
						S_prime, r, is_game_end, _ = self.get_state_data(env, a)
						if r > max_r:
							max_r = r
						if is_game_end:
							break
				r = max_r
				# # give a punishment for failing to finish the game
				# if step==self.steps-1:
				# 	print("fail to Finish")
				# 	r = -1000
				if r > 0:
					r -= step*1.5          # discount for reach checkpoint late

				# Store the experience
				a = q        # use index
				transition = [S, a, r, S_prime, a_prime, is_game_end]
				self.memory.remember(*transition)
				S = S_prime
				a_prime = a

				rewards.append(r)
				total_reward += r
				# print("rewards: ", rewards)
				# print("total_reward: ", total_reward)

				# Generate batch
				batch = self.memory.get_batch_dqlearn(model=self.model, batch_size=self.batch_size, alpha=self.alpha, gamma=self.gamma)

				# Train the network
				if batch:
					inputs, targets = batch
					loss += float(self.model.online_network.train_on_batch(inputs, targets))

				step += 1
				# pbar.update(1)

				# # test
				# if step==10:
					# is_game_end = 1          # game_end restart is effective
				# print("reward: %f , game_state: %d" % (r, is_game_end))
				if is_game_end:
					print("chekpoint achieved! game end!   reward: ", r)          # mannually reset get stuck state also come into here
					# env.reset()
					# env.render()
					# for i in range(88):
					#     S = self.get_state_data(env, [0, 0, 0, 0, 0])    # NOOP until green light
					#     env.render()
					# self.prev_frames = None
					# S, r, is_game_end = self.get_state_data(env, [0, 0, 0, 0, 0])
					break

			# end of while step


			# @amber: now the decay frequency is per episode, to be changed
			# Decay Epsilon
			if self.epsilon > self.final_epsilon and epoch >= self.epislon_wait: 
				self.epsilon -= self.delta_epsilon

			# Decay Alpha
			if self.alpha > self.final_alpha and epoch >= self.alpha_wait: 
				self.alpha -= self.delta_alpha

		

			# save the weights of the last episode
			training_data.append([loss, np.mean(rewards), np.max(rewards), np.min(rewards), np.std(rewards)])
			np.savetxt("data/results/"+ "training_data"+'_' + datetime.now().strftime("%m_%d_%H_%M") + ".csv", np.array(training_data))

			# Save best weights
			total_reward_avg = training_data[-1][1]     # axis 0: 'nb_epoch' dims
			if best_score is None or (best_score is not None and total_reward_avg > best_score):
				self.model.save_weights("model_weigths"+'_'+ datetime.now().strftime("%m_%d_%H_%M") + ".h5")
				best_score = total_reward_avg

		# end of for epoch

		print("%d episodes of training Finished! " % (self.nb_epoch))
		raw_input("Press <enter> to exit... ")
		env.close()


	def test(self, env):

		print("Test:")              
		print("Model:", self.model.__class__.__name__)          # DQN
		print("Algorithm:", self.learn_algo)                    # deep Q-learning
		print("Exploration_Policy:", self.exp_policy)           # epsilon-greedy
		print("Frame Skips:", self.frame_skips)                 # 4
		print("Number of Previous Frames Used:", self.nb_frames)# 3

		env.reset()
		env.render()
		for i in range(88):
		    S, r, is_game_end, _ = self.get_state_data(env, [0, 0, 0, 0, 0])    # NOOP until green light
		    env.render()
		# # skip the straight piece of track, skip the first 2 checkpoints
		# for i in range(160):
		# 	S, r, is_game_end = self.get_state_data(env, [0, 0, 1, 0, 0])
		# 	env.render()

		while is_game_end==0:
			# retrieve action from the model
			Y = self.model.online_network.predict(S)
			q = np.argmax(Y)
			a = self.model.predict(q)
			S, r, is_game_end, _ = self.get_state_data(env, a)
			env.render()			

		raw_input("Press <enter> to exit... ")
		env.close()


	def visualize(self, env, number=100):
		print("visualization:")

		env.reset()
		env.render()
		for i in range(88):
			S, r, is_game_end, _ = self.get_state_data(env, [0, 0, 0, 0, 0])
			env.render()

		# game start
		step = 0
		while is_game_end==0:
			Y = self.model.online_network.predict(S)
			q = np.argmax(Y)
			a = self.model.predict(q)
			S, r, is_game_end, obs = self.get_state_data(env, a)
			env.render()
			if step==number:
				plt.imshow(obs)
				plt.savefig("data/figs/original_fig")
				Y1 = self.model.visualize_network.predict(S)
				Y1 = np.squeeze(Y1, axis=0)
				n, w, h = Y1.shape
				for i in range(n):
					plt.imshow(Y1[i])
					plt.savefig("data/figs/conv1_fig_%d"%(i))
				break
			step += 1
		
		env.close()

class ReplayMemory():

	# used to store transitions and generate batces 

	def __init__(self, memory_size=100):
		self.memory = []
		self._memory_size = memory_size

	def remember(self, s, a, r, s_prime, a_prime, game_over):
		# print("s.shape", s.shape)    # (1, 3, 120, 160)
		self.input_shape = s.shape[1:]
		self.memory.append(np.concatenate([s.flatten(), np.array(a).flatten(), np.array(r).flatten(), s_prime.flatten(), np.array(a_prime).flatten(), 1 * np.array(game_over).flatten()]))
		if self._memory_size > 0 and len(self.memory) > self._memory_size: self.memory.pop(0)

	def get_batch_dqlearn(self, model, batch_size, alpha=1.0, gamma=0.9):

		nb_actions = model.online_network.output_shape[-1]
		# input_shape: (3, 120, 160)
		input_dim = np.prod(self.input_shape)     # prod: multiply every element of the parameter

		# Generate Sample
		if len(self.memory) < batch_size:
			batch_size = len(self.memory)
		samples = np.array(sample(self.memory, batch_size))

		# print("batch_size:", batch_size)
		# print("samples.shape:", samples.shape)

		# Restructure Data------>transition = [S, a, r, S_prime, a_prime, is_game_end]
		S = samples[:, 0 : input_dim]
		a = samples[:, input_dim]            # index
		r = samples[:, input_dim + 1]
		S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2]
		game_over = samples[:, 2 * input_dim + 3]
		r = r.repeat(nb_actions).reshape((batch_size, nb_actions))    # np.repeat: copy n times
		game_over = game_over.repeat(nb_actions).reshape((batch_size, nb_actions))
		S = S.reshape((batch_size, ) + self.input_shape)
		S_prime = S_prime.reshape((batch_size, ) + self.input_shape)
		# print("s.shape", S.shape)    # (batch_size, 3, 120, 160)

		# Predict Q-Values
		X = np.concatenate([S, S_prime], axis=0)
		# print("X.shape", X.shape)    # (2*batch_size, 3, 120, 160)
		Y = model.online_network.predict(X)
		# print("Y.shape", Y.shape)    # (2*batch_size, 16)

		# Get max Q-value
		# Y[batch_size:]----->Q(S_prime)
		Qsa = np.max(Y[batch_size:], axis=1).repeat(nb_actions).reshape((batch_size, nb_actions))
		delta = np.zeros((batch_size, nb_actions))
		a = np.cast['int'](a)     # change the data type to 'int'
		delta[np.arange(batch_size), a] = 1

		# Get target Q-Values
		# Q(s) <- Q(s) + alpha * (r + gamma * max Q(s_prime) - Q(s))
		targets = ((1 - delta) * Y[:batch_size]) + ((alpha * ((delta * (r + (gamma * (1 - game_over) * Qsa))) - (delta * Y[:batch_size]))) + (delta * Y[:batch_size]))
		return S, targets



