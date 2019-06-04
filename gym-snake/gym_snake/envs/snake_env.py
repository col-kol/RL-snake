import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

"""
  Description: 
      Classic snake game, where the goal is direct the snake's head to eat
      an apple (that appears after one has been eating) while the snake grows 
      in length by 1 unit per apple eaten. The snake cannot running into the
      edge of the grid or its trailing body / tail.


  Observation: 


  Action space / Snake direction:

      Up:     [0,  1]    *only if current direction is not Down
      Down:   [0, -1]    *only if current direction is not Up
      Right:  [1,  0]    *only if current direction is not Left
      Left:   [-1, 0]    *only if current direction is not Right


  Reward: 

      Reward is 1 for not eating itself or running into edge.
      Reward is 2 for eating apple.


	Initial state: 

      15 x 15 grid, snake (head) starts at [5, 4] going right, 
      first apple is at [10, 4]


  Encoding state:
  
      - occupancy array of snake: [[x1, y1], [x2, y2], ... [xN, yN]]
          N is the length of snake
      - direction of snake: [x, y] where x and y are in [-1, 1]

      - location of apple: [x, y]

"""
FPS = 30
SCREEN_HEIGHT = 2*256
SCREEN_WIDTH = 2*256

GRID_HEIGHT = 15
GRID_WIDTH = 15

class SnakeEnv(gym.Env):
  metadata = {
              'render.modes': ['human', 'rgb_array'],
              'video.frames_per_second': FPS
  }

  def __init__(self):
  
    self.viewer = None
    
    self.snake_occupancy = [[5, 4], [4, 4], [3, 4]]  #[x, y]
    self.apple_location = [10, 4]  #[x, y]
    self.direction = 0 # going right
    self.apples_eaten = 0

    self.action_space = spaces.Discrete(4) 
    self.done = False
    self.state = None

    self.img_apple = None


  def step(self, action):
    " Process action, then return observation, reward, done, and info for env "

    if action == 0: # Right
      direction = [1, 0]

    if action == 1: # Left
      direction = [-1, 0]

    if action == 2: # Up
      direction = [0, 1]

    if action == 3: # Down
      direction = [0, -1]

    # move snake by direction ie the action
    self.snake_occupancy = self.move_snake([direction])

    head_loc = self.get_snake_occupancy()[0]
    
    # check if the head is out of grid or on its body
    if self.run_into_edge(head_loc) or self.run_into_self(head_loc):
      reward = -1
      return self.get_observation(), reward, True, {} 

    # if snake has eaten apple, extend snake length and generate new apple
    if new_head_loc == self.get_apple_location():

      reward = 2 # update reward

      import random 
      new_x = random.randint(1, GRID_WIDTH)
      new_y = random.randint(1, GRID_HEIGHT)

      self.apple_location = [new_x, new_y]

    
    else:
      # remove last unit of tail  
      self.snake_occupancy.pop()


    if self.check_done_status(self.direction) == True: 
      self.done = True
      reward = 0

    # the snake made a non terminating step so reward
    reward = 1
    obs = self.get_observation()
    

    return obs, reward, False, {}




  def reset(self):
    " Resets environment to initial configuration "

    self.snake_occupancy = [[5, 4], [4, 4], [3, 4]]
    self.apple_location = [10, 4]
    self.direction = 0
    obs = np.array([self.snake_occupancy,
                    self.direction, 
                    self.apple_location])

    return obs




  def render(self, mode='human', close=False):
    
    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(SCREEN_HEIGHT, SCREEN_WIDTH)

      #self.snake = rendering.Line((3., 100.), (100., 100.))
      #self.snake.set_color(.8,.0,.8)
      #self.viewer.add_geom(self.snake)
      from os import path

      fname = path.join(path.dirname(__file__), "red_apple_logo.png")
      self.img_apple = rendering.Image(fname, 100, 100)
      self.viewer.add_onetime(self.img_apple)

    return self.viewer.render(return_rgb_array = mode=='rgb_array')




  def get_observation(self):
    """
        Returns environment observation: 
          an array of snake occupancy, snake direction, and
          location of apple
    """
    snake_occ = self.snake_occupancy
    apple_loc = self.apple_location
    direction = self.direction
    
    obs = np.array([snake_occ, direction, apple_loc])

    return obs



  def check_done_status(self, direction):
    """ 
        Checks whether the environment should be reset if snake runs 
        into itself or edge based on its current direction. 
    """

    return (self.run_into_edge(direction) & self.run_into_self(direction))


  def move_snake(self, direction):
    """
        Moves snake in direction 1 unit and returns the array of grid
        points the new snake occupies
    """
    snake_occ = self.get_snake_occupancy()
    head_loc = snake_occ[0] # snakes head position is first element in occupancy array

    new_head_loc = np.array(head_loc) + np.array(direction) # convert to np array for matrix add
    new_head_loc = new_head_loc.tolist() # convert back to standard array
    
    snake_occ = new_head_loc + snake_occ

    return snake_occ


  def get_snake_occupancy(self):
    " Returns array contains points occupied by snake body"
    return self.snake_occupancy


  def get_apple_location(self):
    " Returns location of apple in form [x, y]"
    return self.apple_location 


  def run_into_edge(self, head_location):
    " Checks if snake is running into the edge. "
    x = head_location[0]
    y = head_location[1]

    return (x > GRID_WIDTH) or (y > GRID_HEIGHT)

  
  def run_into_self(self, head_location):
    " Checks if snake is running into itself. "

    if head_location in self.get_snake_occupancy():
      return True
    else:
      return False


   def close(self):
    " close pyglet viewer "
    if self.viewer:
        self.viewer.close()
        self.viewer = None




