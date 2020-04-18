import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering

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
      N x N grid, snake (head) starts at [5, 4] going right, 
      first apple is at [10, 4]

  Encoding state:
      - occupancy array of snake: [[x1, y1], [x2, y2], ... [xN, yN]]
          N is the length of snake
      - direction of snake: [x, y] where x and y are in [-1, 1]

      - location of apple: [x, y]

"""
FPS = 1
WINDOW_HEIGHT = 500 
WINDOW_WIDTH = 500

UNIT_HEIGHT = 20
UNIT_WIDTH = 20

# snake grid is 25 X 25 "units" 500/20 = 25
GRID_HEIGHT = WINDOW_HEIGHT / UNIT_HEIGHT
GRID_WIDTH = WINDOW_WIDTH / UNIT_WIDTH

class SnakeEnv(gym.Env):
  metadata = {
              'render.modes': ['human', 'rgb_array'],
              'video.frames_per_second': FPS
  }

  def __init__(self):
  
    self.viewer = rendering.Viewer(WINDOW_HEIGHT, WINDOW_WIDTH)
    self.viewer.window.set_caption('SNAKE')

    self.snake_occupancy = None #[x, y]
    self.apple_location = None #[x, y]
    self.direction = None

    self.action_space = spaces.Discrete(4) 
    self.done = False
    self.state = None

  def step(self, action):
    " Process action, then return observation, reward, done, and info for env "
    '''
      We want to learn that actions (directions) will not do anything if the snake 
      is already traveling in that direction or -direction. For example, if the 
      snake is already traveling upwards, pressing Up or Down will not do anything.
      Only pressing Right or Left will.
    '''

    
    ########## process action ##########
    if action == 0: # Right
      new_direction = [1, 0]
      print('Right')
    if action == 1: # Left
      new_direction = [-1, 0]
      print('Left')
    if action == 2: # Up
      new_direction = [0, 1]
      print('Up')
    if action == 3: # Down
      new_direction = [0, -1]
      print('Down')

    opposite_direction = np.multiply(-1, self.direction).tolist() 
    #print('self.direction: ' + str(self.direction))
    #print('opposite_direction: ' + str(opposite_direction))
    #print('new_direction: ' + str(new_direction)+'\n')

    ########## Handle moves that dont change snake direction ##########
    if (new_direction == self.direction) or (new_direction == opposite_direction):

      self.snake_occupancy = self.move_snake([self.direction])
      
      if self.apple_eaten() == True:
        self.apple_location = self.generate_new_apple_loc()
        reward = 1

      else:
        # remove last unit of tail  
        self.snake_occupancy.pop()
        reward = 0
      
      obs = np.array([self.snake_occupancy, self.direction, self.apple_location])

    ########## Handle moves that do change snake direction ##########
    else:
      self.snake_occupancy = self.move_snake([new_direction])
      self.direction = new_direction
      if self.apple_eaten() == True:
        self.apple_location = self.generate_new_apple_loc()
        reward = 1

      else:
        # remove last unit of tail  
        self.snake_occupancy.pop()
        reward = 0
      
      obs = np.array([self.snake_occupancy, self.direction, self.apple_location])
    
    # check is snake run into edge or itself 
    head_loc = self.snake_occupancy[0]
    # check if the head is out of grid or on its body
    if self.run_into_edge(head_loc) or self.run_into_self(self.get_snake_occupancy()):
      reward = -1
      done = True
      if self.run_into_edge(head_loc) == True:
        print('RAN INTO EDGE with head_loc at ' + str(head_loc))

      if self.run_into_self(self.get_snake_occupancy()) == True:
        print('RAN INTO SELF')

      print('RESET')
      return self.get_observation(), reward, True, {} 

    return obs, reward, False, {}

  def reset(self):
    " Resets environment to initial configuration "

    self.snake_occupancy = [[5, 4], [4, 4]]#, [3, 4], [2, 4], [1, 4]]
    self.apple_location = [10, 4]
    self.direction = [1, 0] # moving right
    obs = np.array([self.snake_occupancy, self.direction, self.apple_location])

    return obs

  def render(self, mode='human', close=False):
    
    #self.viewer.window.set_fullscreen()
    self.viewer.window.clear()

    # render apple
    apple = self.viewer.draw_circle(radius=UNIT_WIDTH/4, res=50)
    apple.set_color(1,0,0)
    apple.transform = rendering.Transform()
    x,y = tuple(np.multiply(UNIT_WIDTH, self.get_apple_location()))
    apple.transform.set_translation(x,y)
    apple.add_attr(apple.transform)
    self.viewer.add_onetime(apple)
    
    # render snake  
    snake_occ_arr = self.snake_occupancy
    resized = [np.multiply(UNIT_WIDTH, ele) for ele in snake_occ_arr]
    snake_occ = [tuple(point) for point in resized]

    poly = self.viewer.draw_polyline(snake_occ, color=(0,1,0), linewidth=UNIT_WIDTH)
    self.viewer.add_onetime(poly)

    b,t,r,l = self.render_border()
    b = [tuple(point) for point in b]
    t = [tuple(point) for point in t]
    r = [tuple(point) for point in r]
    l = [tuple(point) for point in l]

    b = self.viewer.draw_polyline(b, color=(0,0,0), linewidth=UNIT_WIDTH)
    self.viewer.add_onetime(b)
    t = self.viewer.draw_polyline(t, color=(0,0,0), linewidth=UNIT_WIDTH)
    self.viewer.add_onetime(t)
    r = self.viewer.draw_polyline(r, color=(0,0,0), linewidth=UNIT_WIDTH)
    self.viewer.add_onetime(r)
    l = self.viewer.draw_polyline(l, color=(0,0,0), linewidth=UNIT_WIDTH)
    self.viewer.add_onetime(l)

    return self.viewer.render(return_rgb_array = mode=='rgb_array')

  def render_border(self):

    bot = [[x, 0] for x in range(0,WINDOW_WIDTH+1,UNIT_WIDTH)]
    top = [[x,WINDOW_HEIGHT] for x in range(0,WINDOW_WIDTH+1,UNIT_WIDTH)]
    l = [[0,y] for y in range(0,WINDOW_HEIGHT+1,UNIT_HEIGHT)]
    r = [[WINDOW_WIDTH,y] for y in range(0,WINDOW_HEIGHT+1,UNIT_HEIGHT)]

    return bot, top, r, l 

  def apple_eaten(self):
    # if snake has eaten apple, extend snake length and generate new apple
    head_loc = self.snake_occupancy[0]
    if head_loc == self.get_apple_location():
      
      print('APPLE EATEN!')
      self.apple_location = self.generate_new_apple_loc()
      #print('new apple loc: '+str(self.apple_location))
      return True

    else:
      return False 

  def generate_new_apple_loc(self):
    " Randomly generates a new apple a [x,y] where [x,y] not where snake is"
    import random

    new_x = random.randint(1, GRID_WIDTH-1)
    new_y = random.randint(1, GRID_HEIGHT-1)

    while [new_x, new_y] in self.snake_occupancy:
      new_x = random.randint(1, GRID_WIDTH-1)
      new_y = random.randint(1, GRID_HEIGHT-1)

    new_apple_loc = [new_x, new_y]  

    return new_apple_loc

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

    return (self.run_into_edge(direction) & self.run_into_self(self.get_snake_occupancy()))

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

    return (x > WINDOW_WIDTH/UNIT_WIDTH -1 ) or (x < 1) or (y > WINDOW_HEIGHT/UNIT_HEIGHT -1) or (y < 1)
  
  def run_into_self(self, snake_occupancy):
    " Checks if snake has run into itself. "
    snake_occ = [tuple(point) for point in snake_occupancy] # convert to tuple for set() 

    return len(snake_occ) != len(set(snake_occ))

  def close(self):
    " close pyglet viewer. "
    if self.viewer:
       self.viewer.close()
       self.viewer = None
