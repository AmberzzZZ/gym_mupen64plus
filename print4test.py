# import gym, gym_mupen64plus

# env = gym.make('Mario-Kart-Luigi-Raceway-v0')

# print("make env")

# env.reset()

# print("reset")

# # # env.render()
# # env.step([0, 0, 0, 0, 0]) 
# # print("step print")

# env.render()

# print("render")
# # print("render print--------")

# env.close()





# import gym, gym_mupen64plus

# env = gym.make('Mario-Kart-Luigi-Raceway-v0')
# # env = gym.make('Mario-Kart-Discrete-Luigi-Raceway-v0')       # more complexed actions

# env.reset()

# # print(env.action_space)
# # # method to show the number of actions
# # print(env.action_space.shape)

# # print(env.observation_space)                  # box(480, 640, 3)
# # print(env.observation_space.shape)            # (480, 640, 3)


# # obs = env._observe()                # private function cannot be called outside the class
# (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) 
# print(obs.shape)                    # (480, 640, 3)


# for i in range(88):
#     (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) 
#     # NOOP until green light
#     env.render()



# for i in range(600):
# 	if i/40 ==0:
# 		env.step([-80, 0, 1, 0, 0])
# 	elif i/80==0:
# 		env.step([0,-80,1,0,0])
# 	else:
# 		env.step([0, 0, 1, 0, 0])
# 	env.render()


# for i in range(100):
# 	# print("start play")
#     (obs, rew, end, info) = env.step([0, 0, 1, 0, 0]) 
#     # Drive straight
#     env.render()

# raw_input("Press <enter> to exit... ")

# env.close()


import numpy as np

# a = np.array([
# 				[[1, 2], [3,4],
# 				[1, 1], [1, 6]],

# 				[[1, 2], [1,1],
# 				 [3,1], [1,2]],

# 				])

# print(a.shape)

# print(a.dot([1,2]))


a = np.ones((3, 480, 640))
print(a.shape)


print(a.dot([1,2]))