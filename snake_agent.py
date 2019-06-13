import gym
import gym_snake


env = gym.make('snake-v0')

observation = env.reset()

for _ in range(1):
	done = False
	while(not done):

		env.render()
		action = env.action_space.sample()

		#input('\nPress Enter')
		#print('action: ' +str(action))
		observation, reward, done, info = env.step(action)
		#env.render()
		#print('reward: ' + str(reward))
		#print('done status: '+ str(done))
		#print('snake_occupancy: ' + str(observation[0]) +'\n')

		import time
		time.sleep(.5)
	
	env.render() # render terminating move/state

	time.sleep(10)
#env.close()
