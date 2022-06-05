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

    def __init__(self, model_path):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 512, 3)
        self.model.load_state_dict(torch.load(model_path))


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)


def visualise():
    game = SnakeGameAI()
    models = os.listdir('model')
    models = sorted(models, key=lambda model: int(model.split()[1].replace(".pth", "")))

    for model_path in models:

        agent = Agent(os.path.join('model', model_path))
        while True:
            
            game.model_display_name = model_path.replace('.pth', '')
            # get old state
            state_old = torch.tensor(agent.get_state(game)).type(torch.FloatTensor)

            # get move
            final_pred = agent.model(state_old)

            final_move = np.zeros(3, dtype=int)
            final_move[torch.argmax(final_pred)] = 1

            # perform move and get new state
            _, done, _ = game.play_step(final_move)

            if done:
                # train long memory, plot result
                game.reset()
                break