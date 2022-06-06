"""
On récupère des modèles de différentes versions pour les visualiser
"""

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
#font = pygame.font.Font('./snake_game/arial.ttf', 25)
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

W=640
H=480

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40

class Snake:

    def __init__(self, name:int):
        self.name = name
        self.reward = 0
        self.head = Point((np.random.randint(0,W)%20)*20, (np.random.randint(0,H)%20)*20)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.direction = Direction.RIGHT

    
    def move(self, action):
        
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


class SnakeGameAI:

    def __init__(self, nb_snake = 1):
        self.w = W
        self.h = H
        self.snakes = []
        self.nb_snake = nb_snake
        self.model_display_name = ""
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def _create_snakes(self):

        for i in range(self.nb_snake):
            self.snakes.append(Snake(i))

    def reset(self):
        # init game state
        self.snakes = []
        self._create_snakes()

        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        for snake in self.snakes:
            if self.food in snake.snake:
                self._place_food()


    def play_step(self, actions):

        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            

        for snake, action in zip(self.snakes, actions):

            # 2. move
            snake.move(action) # update the head
            snake.snake.insert(0, snake.head)
            
            # 3. check if game over
            snake.reward = 0
            game_over = False

            if self.is_collision(snake) or self.frame_iteration > 100*len(snake.snake):
                game_over = True
                snake.reward = -10
                return game_over

            # 4. place new food or just move
            if snake.head == self.food:
                snake.reward = 10
                snake.score += 1
                self._place_food()
            else:
                snake.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over


    def is_collision(self, my_snake, pt=None):
        if pt is None:
            pt = my_snake.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # hits itself
        if pt in my_snake.snake[1:]:
            return True

        # hits other snakes
        for snake in self.snakes:
            if(snake.name != my_snake.name):
                if pt in snake.snake:
                    return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)
        for snake in self.snakes:
            for pt in snake.snake:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

            pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        name = font.render(self.model_display_name, True, WHITE)
        self.display.blit(name, [0, 0])
        pygame.display.flip()