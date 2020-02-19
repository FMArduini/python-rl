import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import os
import time

MAX_STEPS = 250

SNAKE_VALUE = 2
WALL_VALUE = 1
EMPTY_VALUE = 0
FOOD_VALUE = 3

YDIM, XDIM = 15, 15

ACTION_SET = {
    0: np.array([-1, 0]),  # up
    1: np.array([1, 0]),  # down
    2: np.array([0, 1]),  # right
    3: np.array([0, -1])  # left
}


def ind(a, b, c=None):
    if c:
        a[b[0], b[1]] = c
    else:
        return a[b[0], b[1]]


class Snake:
    def __init__(self, head):
        self.head = head
        self.body = [head, self.head + np.array([1, 0]), self.head + np.array([2, 0])]

    def move_body(self, direction, keep_tail=False):
        # if we want to keep the tail we keep the body as it is.
        # otherwise we remove the last entry of the body
        if not keep_tail:
            self.body = self.body[:-1]

        # we insert at the start of the body the new head's position
        self.body.insert(0, self.head + direction)
        # we update the position of the head
        self.head = self.body[0]


class Board:
    def __init__(self, xdim, ydim):
        self.xdim = xdim
        self.ydim = ydim
        self.board = np.zeros(shape=(self.ydim, self.xdim))

        self.food_cords = None  # np.array([10,15])# None

    def create_walls(self):
        # walls
        self.board[:, 0] = WALL_VALUE
        self.board[:, -1] = WALL_VALUE
        self.board[0, :] = WALL_VALUE
        self.board[-1, :] = WALL_VALUE

    def refresh(self, snake):
        self.board = np.zeros(shape=(self.ydim, self.xdim))

        self.create_walls()
        self.place_snake(snake)
        self.place_food()

    def place_food(self):
        # print(self.food_cords,type(self.food_cords))

        if isinstance(self.food_cords, type(None)):
            self.create_food()
        self.set_value(self.food_cords, FOOD_VALUE)

    def set_value(self, index_array, value):
        self.board[index_array[0], index_array[1]] = value

    def get_value(self, index_array):
        return self.board[index_array[0], index_array[1]]

    def place_snake(self, snake):
        for i in snake.body:
            self.set_value(index_array=i, value=SNAKE_VALUE)

    def get_state(self):
        return self.board

    def delete_food(self):
        self.food_cords = None

    def create_food(self):
        zeros = np.argwhere(self.board == 0)
        self.food_cords = list(zeros)[np.random.randint(0, zeros.shape[0])]


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.steps = None
        self.done = None
        self.reward = None
        self.direction = None
        self.board = None
        self.snake = None
        self.next_head_pos = None
        self.next_head_pos_value = None

        self.t_2 = None
        self.t_1 = None
        self.t_0 = None

        self.action_space = spaces.Discrete(4)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=3,
                                            shape=(XDIM, YDIM, 3), dtype=np.uint8)

        self.reset()
        print("Snake RL v: 1.07")

    def reset(self):
        self.steps = 0
        self.done = 0
        self.reward = 0

        self.direction = np.array([-1, 0])  # going up

        half = YDIM // 2

        self.board = Board(xdim=XDIM, ydim=YDIM)

        # snake
        head_position = np.array([half, half])
        self.snake = Snake(head_position)
        self.board.refresh(self.snake)

        self.t_0 = self.board.get_state()
        self.t_1 = self.t_0
        self.t_2 = self.t_1

        return self.get_state()

    def update_routine(self):
        # this routine will check all rules and update the board
        self.next_head_pos = self.snake.head + self.direction
        self.next_head_pos_value = self.board.get_value(self.next_head_pos)
        game_over, keep_tail = self.check_move()

        self.reward = -0.01  # reward for nothing happening

        if game_over:
            self.reward = 0  # reward for dying
            return True

        if keep_tail:
            self.board.delete_food()
            self.reward = 10

        self.snake.move_body(self.direction, keep_tail=keep_tail)
        self.board.refresh(self.snake)

        self.t_2 = self.t_1
        self.t_1 = self.t_0
        self.t_0 = self.board.get_state()

        return False

    def check_move(self):
        if np.equal(self.next_head_pos, self.snake.body[-1]).all():
            # if new head is where the last (about to be deleted) tail is then its okay.
            return False, False
        if self.next_head_pos_value == WALL_VALUE or self.next_head_pos_value == SNAKE_VALUE:
            # if next pos value is wall or snake it game over.
            return True, False
        elif self.next_head_pos_value == FOOD_VALUE:
            # ate food
            return False, True
        elif self.next_head_pos_value == EMPTY_VALUE:
            # its empty
            return False, False
        else:
            raise Exception('what?')

    def apply_action(self, action):
        if np.equal(ACTION_SET[action], self.direction).all():
            # if the new dir is the same as the old one, we dont apply a change.
            return False
        elif np.equal(ACTION_SET[action] + self.direction, np.array([0, 0])).all():
            # if the action + the prev direction is 0,0 then its the inverse (up + down) (left+right)
            # we dont apply
            return False
        else:
            # otherwise we update the direction with the new one.
            self.direction = ACTION_SET[action]
            return True

    def get_state(self):
        return np.dstack([self.t_0, self.t_1, self.t_2])

    def step(self, action):
        self.steps += 1
        # recive an action and change direction
        self.apply_action(action)
        # move
        game_over = self.update_routine()
        done = True if game_over or self.steps >= MAX_STEPS else False
        return self.get_state(), self.reward, done, {}


    def render(self, mode='human'):
        print("\n" * 100)
        for r in self.board.board:
            row = ' '.join([str(int(i)) for i in r])
            print(row)
