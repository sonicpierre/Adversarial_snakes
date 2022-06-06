import torch
import numpy as np
import os
from collections import deque
from visualise_model.game_vis import SnakeGameAI, Direction, Point
from snake_pygame.model import Linear_QNet

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self, model_path, snake):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 512, 3)
        self.model.load_state_dict(torch.load(model_path))
        self.snake = snake

    def get_state(self, game):

        head = self.snake.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = self.snake.direction == Direction.LEFT
        dir_r = self.snake.direction == Direction.RIGHT
        dir_u = self.snake.direction == Direction.UP
        dir_d = self.snake.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(self.snake, pt = point_r)) or 
            (dir_l and game.is_collision(self.snake, pt = point_l)) or 
            (dir_u and game.is_collision(self.snake, pt = point_u)) or 
            (dir_d and game.is_collision(self.snake, pt = point_d)),

            # Danger right
            (dir_u and game.is_collision(self.snake, pt = point_r)) or 
            (dir_d and game.is_collision(self.snake, pt = point_l)) or 
            (dir_l and game.is_collision(self.snake, pt = point_u)) or 
            (dir_r and game.is_collision(self.snake, pt = point_d)),

            # Danger left
            (dir_d and game.is_collision(self.snake, pt = point_r)) or 
            (dir_u and game.is_collision(self.snake, pt = point_l)) or 
            (dir_r and game.is_collision(self.snake, pt = point_u)) or 
            (dir_l and game.is_collision(self.snake, pt = point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < self.snake.head.x,  # food left
            game.food.x > self.snake.head.x,  # food right
            game.food.y < self.snake.head.y,  # food up
            game.food.y > self.snake.head.y  # food down
            ]

        return np.array(state, dtype=int)


def visualise():

    nb_snake = 1
    game = SnakeGameAI(nb_snake)
    models_dir = os.listdir('model')
    models_dir = sorted(models_dir, key=lambda model: int(model))
    models = []
    for dir in models_dir:
        dir_path = os.path.join('model', dir)
        path = os.listdir(dir_path)
        models.append(os.path.join(dir_path,np.random.choice(path)))

    agents = []
    for snake, model_path in zip(game.snakes, models):
        agents.append(Agent(model_path, snake))


    while True:

        actions = []
        for agent in agents:
            # get old state
            state_old = agent.get_state(game)
        
            # get old state
            state_old = torch.tensor(agent.get_state(game)).type(torch.FloatTensor)

            # get move
            final_pred = agent.model(state_old)

            final_move = np.zeros(3, dtype=int)
            final_move[torch.argmax(final_pred)] = 1
            actions.append(final_move)

        # perform move and get new state
        done = game.play_step(actions)

        if done:
            game.reset()
            break