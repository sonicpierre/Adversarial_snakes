import random
import numpy as np
import config


class Snake:
    def __init__(self, name: int):
        self.name = name
        self.reward = 0
        self.head = config.Point(
            (np.random.randint(20, config.W - 20) % 20) * 20,
            (np.random.randint(20, config.H - 20) % 20) * 20,
        )
        self.snake = [
            self.head,
            config.Point(self.head.x - config.BLOCK_SIZE, self.head.y),
            config.Point(self.head.x - (2 * config.BLOCK_SIZE), self.head.y),
        ]
        self.score = 0
        self.direction = config.Direction.RIGHT

    def move(self, action):

        # [straight, right, left]
        clock_wise = [
            config.Direction.RIGHT,
            config.Direction.DOWN,
            config.Direction.LEFT,
            config.Direction.UP,
        ]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == config.Direction.RIGHT:
            x += config.BLOCK_SIZE
        elif self.direction == config.Direction.LEFT:
            x -= config.BLOCK_SIZE
        elif self.direction == config.Direction.DOWN:
            y += config.BLOCK_SIZE
        elif self.direction == config.Direction.UP:
            y -= config.BLOCK_SIZE

        self.head = config.Point(x, y)


class SnakeGameAI:
    def __init__(self, nb_snake=1):
        self.w = config.W
        self.h = config.H
        self.nb_snake = nb_snake
        self.snakes = []
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
        x = (
            random.randint(0, (self.w - config.BLOCK_SIZE) // config.BLOCK_SIZE)
            * config.BLOCK_SIZE
        )
        y = (
            random.randint(0, (self.h - config.BLOCK_SIZE) // config.BLOCK_SIZE)
            * config.BLOCK_SIZE
        )
        self.food = config.Point(x, y)
        for snake in self.snakes:
            if self.food in snake.snake:
                self._place_food()

    def play_step(self, actions):
        self.frame_iteration += 1

        for snake, action in zip(self.snakes, actions):

            # 2. move
            snake.move(action)  # update the head
            snake.snake.insert(0, snake.head)

            # 3. check if game over
            snake.reward = 0
            game_over = False

            # check different collision
            collision = self.is_collision(snake)
            if collision == 1 or self.frame_iteration > 100 * len(snake.snake):
                game_over = True
                snake.reward = -10
                return game_over

            elif collision == 2:
                game_over = True
                snake.reward = -5
                return game_over

            # 4. place new food or just move
            if snake.head == self.food:
                snake.reward = 15
                snake.score += 1
                self._place_food()
            else:
                snake.snake.pop()

        # 6. return game over
        return game_over

    def is_collision(self, my_snake, pt=None) -> int:
        if pt is None:
            pt = my_snake.head

        # hits boundary
        if (
            pt.x > self.w - config.BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - config.BLOCK_SIZE
            or pt.y < 0
        ):
            return 1

        # hits itself
        if pt in my_snake.snake[1:]:
            return 1

        # hits other snakes
        for snake in self.snakes:
            if snake.name != my_snake.name:
                if pt in snake.snake:
                    return 2

        return 0
