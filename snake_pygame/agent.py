import torch
import random
import numpy as np
from collections import deque
from snake_pygame.game import SnakeGameAI, Direction, Point
from snake_pygame.model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self, snake):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 512, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGameAI(nb_snake=1)
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
        done, score = game.play_step(actions)
        
        for agent, action in zip(agents, actions):
            state_new = agent.get_state(game)
            # train short memory
            agent.train_short_memory(state_old, action, agent.snake.reward, state_new, done)

            # remember
            agent.remember(state_old, action, agent.snake.reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            for snake, agent in zip(game.snakes, agents):
                agent.snake = snake

            for agent in agents:
                agent.n_games += 1
                agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(version = agent.n_games)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            print("Score :",score)
            print("Mean score :", mean_score)