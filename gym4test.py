import gym

env = gym.make('CartPole-v0')
env.reset()

print(env.action_space)

for i in range(10):
	print(env.action_space.sample())