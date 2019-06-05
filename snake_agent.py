import gym
import gym_snake


env = gym.make('snake-v0')

observation = env.reset()

for _ in range(1):
	done = False
	while(not done):

		env.render()
		action = env.action_space.sample()
		#action = 4
		input('\nPress Enter')
		#print('action: ' +str(action))
		observation, reward, done, info = env.step(action)
	
		#print('reward: ' + str(reward))
		print('done status: '+ str(done))
		#print('snake_occupancy: ' + str(observation[0]) +'\n')

		import time
		#time.sleep(1)

	time.sleep(3)
env.close()
