import gym
import gym_snake

env = gym.make('snake-v0')

observation = env.reset()

for _ in range(100):

	env.render()
	action = env.action_space.sample()
	#print(action)
	observation, reward, done, info = env.step(action)
	

env.close()
