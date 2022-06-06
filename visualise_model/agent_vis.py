import torch
import numpy as np
import os
import config as config
from collections import deque
from visualise_model.game_vis import SnakeGameAI
from snake_pygame.model import Linear_QNet


class Agent:
    def __init__(self, model_path, snake):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.90  # discount rate
        self.memory = deque(maxlen=config.MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, [128], 3)
        self.model.load_state_dict(torch.load(model_path))
        self.snake = snake

    def get_state(self, game):

        head = self.snake.snake[0]
        point_l = config.Point(head.x - 20, head.y)
        point_r = config.Point(head.x + 20, head.y)
        point_u = config.Point(head.x, head.y - 20)
        point_d = config.Point(head.x, head.y + 20)

        dir_l = self.snake.direction == config.Direction.LEFT
        dir_r = self.snake.direction == config.Direction.RIGHT
        dir_u = self.snake.direction == config.Direction.UP
        dir_d = self.snake.direction == config.Direction.DOWN

        state = [
            # Danger straight
            (dir_r and not game.is_collision(self.snake, pt=point_r))
            or (dir_l and not game.is_collision(self.snake, pt=point_l))
            or (dir_u and not game.is_collision(self.snake, pt=point_u))
            or (dir_d and not game.is_collision(self.snake, pt=point_d)),
            # Danger right
            (dir_u and not game.is_collision(self.snake, pt=point_r))
            or (dir_d and not game.is_collision(self.snake, pt=point_l))
            or (dir_l and not game.is_collision(self.snake, pt=point_u))
            or (dir_r and not game.is_collision(self.snake, pt=point_d)),
            # Danger left
            (dir_d and not game.is_collision(self.snake, pt=point_r))
            or (dir_u and not game.is_collision(self.snake, pt=point_l))
            or (dir_r and not game.is_collision(self.snake, pt=point_u))
            or (dir_l and not game.is_collision(self.snake, pt=point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < self.snake.head.x,  # food left
            game.food.x > self.snake.head.x,  # food right
            game.food.y < self.snake.head.y,  # food up
            game.food.y > self.snake.head.y,  # food down
        ]

        return np.array(state, dtype=int)


def visualise(nb_snake=1):

    game = SnakeGameAI(nb_snake)
    models_dir = os.listdir("model")
    models_dir = sorted(models_dir, key=lambda model: int(model))

    # Look for the best model in the models folders
    models = []
    for dir in models_dir:
        dir_path = os.path.join("model", dir)
        paths = os.listdir(dir_path)
        paths = sorted(paths, key=lambda path: int(path.split()[1].replace(".pth", "")))
        models.append(os.path.join(dir_path, paths[-1]))

    # Create agents
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

            # standardise the action
            final_move = np.zeros(3, dtype=int)
            final_move[torch.argmax(final_pred)] = 1
            actions.append(final_move)

        # perform move and get new state
        done = game.play_step(actions)

        if done:
            game.reset()
            agents = []
            for snake, model_path in zip(game.snakes, models):
                agents.append(Agent(model_path, snake))
