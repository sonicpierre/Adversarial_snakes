import torch
import random
import numpy as np
import config
import os
from collections import deque
from snake_pygame.game import SnakeGameAI
from snake_pygame.model import Linear_QNet, QTrainer


class Agent:
    def __init__(self, snake):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.90  # discount rate
        self.memory = deque(maxlen=config.MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, [128], 3)
        self.trainer = QTrainer(self.model, lr=config.LR, gamma=self.gamma)
        self.snake = snake

    def get_state(self, game):
        head = self.snake.snake[0]
        point_l = config.Point(head.x - config.BLOCK_SIZE, head.y)
        point_r = config.Point(head.x + config.BLOCK_SIZE, head.y)
        point_u = config.Point(head.x, head.y - config.BLOCK_SIZE)
        point_d = config.Point(head.x, head.y + config.BLOCK_SIZE)

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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > config.BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, config.BATCH_SIZE
            )  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 120 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(nb_snake=1):

    record = np.zeros(nb_snake)
    game = SnakeGameAI(nb_snake)

    agents = []
    for snake in game.snakes:
        agents.append(Agent(snake))

    while True:

        actions = []
        for agent in agents:
            # get old state
            state_old = agent.get_state(game)

            # get move
            action = agent.get_action(state_old)
            actions.append(action)

        # perform move and get new state
        done = game.play_step(actions)

        for agent, action in zip(agents, actions):
            state_new = agent.get_state(game)
            # train short memory
            agent.train_short_memory(
                state_old, action, agent.snake.reward, state_new, done
            )

            # remember
            agent.remember(state_old, action, agent.snake.reward, state_new, done)

        if done:

            for agent in agents:
                agent.n_games += 1
                agent.train_long_memory()
                amelioration = False
                if agent.snake.score > record[agent.snake.name]:
                    path_model_dir = os.path.join("model", str(agent.snake.name))
                    if not os.path.exists(path_model_dir):
                        os.mkdir(path_model_dir)
                    agent.model.save(
                        model_folder_path=path_model_dir, version=agent.snake.score
                    )
                    amelioration = True

                if amelioration:
                    record[agent.snake.name] = agent.snake.score

            print("Game", agent.n_games, "Record:", record)

            game.reset()

            # train long memory, plot result
            for snake, agent in zip(game.snakes, agents):
                agent.snake = snake
