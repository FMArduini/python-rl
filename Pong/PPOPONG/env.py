
# coding: utf-8

# In[1]:


# PONG pygame


import numpy as np
import pygame
from pygame.locals import *
import pickle
import warnings

import tensorflow as tf
import numpy as np
import os

# colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


class Game:
    def __init__(self, game_length=None, game_time_limit=None):
        self.globals = Globals()
        self.game_length = game_length
        self.game_time_limit = game_time_limit
        # initiate ball and player paddles
        self.ball = Ball(self)
        self.paddle1, self.paddle2 = self.init_paddles()

        # trackers and variables
        self.score1 = self.score2 = 0
        self.saves = 0
        self.paddle1moves = []
        self.paddle2moves = []

        self.t = 0  # timestep

        # game log tracks the position of paddles and ball at each frame so we can recreate.
        self.game_log = []

    def check_runtime(self):
        if not self.game_length and not self.game_time_limit:
            warnings.warn("No ending mechanic specified, game will run for 10 games")
            self.game_length = 10

    def game_end(self):
        if self.game_time_limit:
            if self.t >= self.game_time_limit:
                return True
            else:
                return False

        elif self.game_length:
            if self.paddle1.score >= self.game_length or self.paddle2.score >= self.game_length:
                return True

            elif self.saves > self.globals.n_save_stop:
                # draw; dont let it go on forever
                self.paddle1.score += 1
                self.paddle2.score += 1
                return True
            else:
                return False

    def get_input(self):
        return np.array([
            self.ball.x,
            self.ball.y,
            self.ball.vx,
            self.ball.vy,
            self.paddle1.x,
            self.paddle1.y
        ])

    def log(self, fitness=None):
        self.game_log.append(
            {'ballx': self.ball.x,
             'bally': self.ball.y,
             'paddle1x': self.paddle1.x,
             'paddle1y': self.paddle1.y,
             'paddle2x': self.paddle2.x,
             'paddle2y': self.paddle2.y,
             'paddle1score': self.paddle1.score,
             'paddle2score': self.paddle2.score,
             'fitness': fitness}
        )

    def save_game(self, filepath):
        pickle.dump(self, open(filepath, 'wb'))

    def display_frame(self, window, frame):
        self.display_field(window)
        pygame.draw.circle(window, RED, [frame['ballx'], frame['bally']], self.ball.radius, 0)

        half_pad_width = self.globals.HALF_PAD_WIDTH
        half_pad_height = self.globals.HALF_PAD_HEIGHT
        pygame.draw.polygon(window, GREEN, [[frame['paddle1x'] - half_pad_width, frame['paddle1y'] - half_pad_height],
                                            [frame['paddle1x'] - half_pad_width, frame['paddle1y'] + half_pad_height],
                                            [frame['paddle1x'] + half_pad_width, frame['paddle1y'] + half_pad_height],
                                            [frame['paddle1x'] + half_pad_width, frame['paddle1y'] - half_pad_height]],
                            0)

        pygame.draw.polygon(window, GREEN, [[frame['paddle2x'] - half_pad_width, frame['paddle2y'] - half_pad_height],
                                            [frame['paddle2x'] - half_pad_width, frame['paddle2y'] + half_pad_height],
                                            [frame['paddle2x'] + half_pad_width, frame['paddle2y'] + half_pad_height],
                                            [frame['paddle2x'] + half_pad_width, frame['paddle2y'] - half_pad_height]],
                            0)

    def replay(self, canvas_name=None):
        pygame.init()

        # canvas declaration
        window = pygame.display.set_mode((self.globals.WIDTH, self.globals.HEIGHT), 0, 32)
        pygame.display.set_caption('REPLAY' if not canvas_name else canvas_name)

        clock = pygame.time.Clock()

        for frame in self.game_log:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
            self.display_frame(window, frame)
            pygame.display.update()
            clock.tick(60)
        pygame.quit()
        quit()

    def init_paddles(self):
        paddle1 = Paddle(self, self.globals.HALF_PAD_WIDTH - 1, self.globals.HEIGHT // 2)
        paddle2 = Paddle(self, self.globals.WIDTH + 1 - self.globals.HALF_PAD_WIDTH, self.globals.HEIGHT // 2)
        return paddle1, paddle2

    def update(self, log=False):
        if log: self.log()
        self.ball.move()
        self.paddle1.move()
        self.paddle2.move()
        self.t += 1
        return self.game_end()

    def display_field(self, canvas):
        canvas.fill(BLACK)
        pygame.draw.line(canvas, WHITE, [self.globals.WIDTH // 2, 0], [self.globals.WIDTH // 2, self.globals.HEIGHT], 1)
        pygame.draw.line(canvas, WHITE, [self.globals.PAD_WIDTH, 0], [self.globals.PAD_WIDTH, self.globals.HEIGHT], 1)
        pygame.draw.line(canvas, WHITE, [self.globals.WIDTH - self.globals.PAD_WIDTH, 0],
                         [self.globals.WIDTH - self.globals.PAD_WIDTH, self.globals.HEIGHT], 1)
        pygame.draw.circle(canvas, WHITE, [self.globals.WIDTH // 2, self.globals.HEIGHT // 2], 70, 1)
        pygame.draw.circle(canvas, WHITE, [self.globals.WIDTH // 2, self.globals.HEIGHT // 2], 70, 1)

        myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
        label1 = myfont1.render("NEAT Score " + str(self.paddle1.score), 1, (255, 255, 0))
        canvas.blit(label1, (50, 20))

        myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
        label2 = myfont2.render("Computer Score " + str(self.paddle2.score), 1, (255, 255, 0))
        canvas.blit(label2, (400, 20))

    def display_centre_text(self, canvas, text):
        myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
        label3 = myfont2.render(text, 1, (255, 255, 0))
        canvas.blit(label3, (self.globals.WIDTH * 0.385, self.globals.HEIGHT * 0.9))

    def display(self, canvas, seconds=None):
        # display field
        self.display_field(canvas)
        # display ball
        self.ball.display(canvas)
        # display paddles
        self.paddle1.display(canvas)
        self.paddle2.display(canvas)

    def display_seconds(self, canvas, seconds):
        myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
        label3 = myfont2.render(f'{seconds:04}', 1, (255, 255, 0))
        canvas.blit(label3, (self.globals.WIDTH * 0.45, self.globals.HEIGHT * 0.05))

    def computer_move(self, computer_speed):
        # move the second paddle towards the ball
        if self.ball.y < self.paddle2.y:
            self.paddle2.vy = -1 * computer_speed
        elif self.ball.y >= self.paddle2.y:
            self.paddle2.vy = 1 * computer_speed

    def paddle1move(self, action, speed):
        self.paddle1.vy = action * speed
        self.paddle1moves.append(action)
        
    def paddle2move(self, action, speed):
        self.paddle2.vy = action * speed
        self.paddle2moves.append(action)

    def keydown(self, event):
        if event.key == K_UP:
            self.paddle1.vy = -8
        elif event.key == K_DOWN:
            self.paddle1.vy = 8
        elif event.key == K_w:
            self.paddle2.vy = -8
        elif event.key == K_s:
            self.paddle2.vy = 8

    # keyup handler
    def keyup(self, event):
        if event.key in (K_w, K_s):
            self.paddle2.vy = 0
        elif event.key in (K_UP, K_DOWN):
            self.paddle1.vy = 0


class Ball:
    def __init__(self, game):
        self.game = game
        self.globals = self.game.globals

        self.radius = self.globals.BALL_RADIUS

        self.speed = self.globals.SPEED
        self.x = self.y = self.vx = self.vy = None

        self.increase_speed = False
        self.reset()

    def reset(self, direction='right'):
        self.y = np.random.randint((self.radius * 2), self.globals.HEIGHT - (self.radius * 2))
        self.speed = self.globals.SPEED
        self.increase_speed = False

        self.game.saves = 0

        self.vy = np.random.choice([-1, 1]) * np.random.choice(np.arange(1, 3, 0.01))
        if direction == 'left':
            self.x = int(self.globals.WIDTH * 0.95)
            self.vx = -1 * self.globals.SPEED
        elif direction == 'right':
            self.x = int(self.globals.WIDTH * 0.05)
            self.vx = 1 * self.globals.SPEED
        else:
            self.vx = np.random.choice([-1, 1]) * self.globals.SPEED

    def check_speed(self):
        r = np.sqrt((self.vx ** 2) + (self.vy ** 2)) / self.speed
        if self.vy > 0 and self.vy < 1:
            self.vy = 1
        elif self.vy < 0 and self.vy > -1:
            self.vy = -1
        self.vx = self.vx * 1 / np.sqrt(r)
        self.vy = self.vy * 1 / np.sqrt(r)
        pass

    def move(self):
        # make sure speed is constant

        self.check_speed()

        self.x = int(self.x + self.vx)
        self.y = int(self.y + self.vy)
        self.check_collision()
        if self.increase_speed:
            self.speed += 0.5
            self.increase_speed = False

    def bounce(self, diff):
        if abs(diff) > 0.9 * self.globals.HALF_PAD_HEIGHT:
            self.vy += 1
        elif abs(diff) > 0.975 * self.globals.HALF_PAD_HEIGHT:
            self.vy += 2

        if diff < 0:
            self.vy = abs(self.vy)
        if diff > 0:
            self.vy = abs(self.vy) * -1

        pass

    def check_collision(self):
        # ball collision check on top and bottom walls
        if self.y <= self.radius:
            self.vy = -1 * self.vy  # invert vy
            self.y = self.radius + 1
        if self.y >= self.globals.HEIGHT - self.radius:
            self.vy = -1 * self.vy  # invert vy

        # PADDLE 1 LEFT PADDLE
        if self.x - self.radius <= self.game.paddle1.x + self.globals.HALF_PAD_WIDTH and                 self.y in range(self.game.paddle1.y - self.globals.HALF_PAD_HEIGHT,
                                self.game.paddle1.y + self.globals.HALF_PAD_HEIGHT - 1,
                                1):

            self.x = self.globals.PAD_WIDTH + self.radius

            diff = self.game.paddle1.y - self.y
            self.bounce(diff)
            self.vx = -1 * self.vx
            self.game.paddle1.saves += 1
            self.game.saves += 1
            self.increase_speed = True

        elif self.x <= self.radius + self.game.paddle1.x + self.globals.HALF_PAD_WIDTH:
            self.game.paddle2.score += 1
            self.reset(direction='left')

        # paddle 2 RIGHT PADDLE
        if self.x + self.radius >= self.globals.WIDTH + 1 - self.globals.HALF_PAD_WIDTH and                 self.y in range(self.game.paddle2.y - self.globals.HALF_PAD_HEIGHT,
                                self.game.paddle2.y + self.globals.HALF_PAD_HEIGHT - 1,
                                1):

            self.x = self.globals.WIDTH - self.globals.PAD_WIDTH - self.radius

            diff = self.game.paddle2.y - self.y
            self.bounce(diff)
            self.vx = -1 * self.vx
            self.game.paddle2.saves += 1
            self.game.saves += 1
            self.increase_speed = True

        elif self.x + self.radius >= self.globals.WIDTH + 1 - self.globals.HALF_PAD_WIDTH:
            self.game.paddle1.score += 1
            self.reset(direction='left')

    def display(self, canvas):
        pygame.draw.circle(canvas, RED, [self.x, self.y], self.radius, 0)


class Paddle:
    def __init__(self, game, x, y):
        self.game = game
        self.globals = self.game.globals

        self.pad_width = self.globals.PAD_WIDTH
        self.pad_height = self.globals.PAD_HEIGHT
        self.half_pad_width = self.globals.HALF_PAD_WIDTH
        self.half_pad_height = self.globals.HALF_PAD_HEIGHT

        self.x = x
        self.y = y

        self.vy = 0

        self.score = 0
        self.saves = 0

        self.position_log = []

    def move(self):
        if self.vy == 0:
            return None

        newy = self.y + self.vy

        if newy > self.globals.HEIGHT - self.half_pad_height:
            # above out of field
            newy = self.globals.HEIGHT - self.half_pad_height
        elif newy < 0 + self.half_pad_height:
            newy = 0 + self.half_pad_height

        self.y = newy

        self.position_log.append({'x': self.x, 'y': self.y})

    def display(self, canvas):
        pygame.draw.polygon(canvas, GREEN, [[self.x - self.half_pad_width, self.y - self.half_pad_height],
                                            [self.x - self.half_pad_width, self.y + self.half_pad_height],
                                            [self.x + self.half_pad_width, self.y + self.half_pad_height],
                                            [self.x + self.half_pad_width, self.y - self.half_pad_height]], 0)


# globals
class Globals:
    n_save_stop = 30
    WIDTH = 600
    HEIGHT = 400
    BALL_RADIUS = 20
    PAD_WIDTH = 8
    PAD_HEIGHT = 40
    HALF_PAD_WIDTH = PAD_WIDTH // 2
    HALF_PAD_HEIGHT = PAD_HEIGHT // 2
    CENTRE = [HALF_PAD_WIDTH, HALF_PAD_HEIGHT]

    SPEED = 5


# In[2]:


class Env():
    def __init__(self, game_length=10, pcspeed=3, state_lookback = 3):
        self.game_length = game_length
        self.computer_speed = pcspeed
        self.player_speed = pcspeed
        self.state_lookback = state_lookback
        self.reset()
    
    
    
    def reset(self):
        self.actions_taken = []
        self.game = Game(game_length=self.game_length)
        self.step_n = 0
        self.save_count = self.previous_save_count = 0
        self.score_count = self.previous_score_count = 0
        self.r = 0
        self.__reset_state_memory()
        return self.get_state()
        
    def __reset_state_memory(self):
        s = self.__get_current_state()
        shape = s.shape
        self.state_memory = np.zeros((s.shape[0],self.state_lookback))
        self.state_memory[:,0] = s
        
    def get_state(self):
        return self.state_memory.flatten()[None,:]
    
    def __update_state_memory(self):
        state = self.__get_current_state()
        self.state_memory[:,1:] = self.state_memory[:,:-1]
        self.state_memory[:,0] = state
        
    def check_save(self):
        self.save_count = self.game.paddle1.saves
        if self.save_count > self.previous_save_count:
            self.previous_save_count = self.save_count
            return True
        else:
            return False
        
    def __get_current_state(self):
        return np.array([
            self.game.ball.x,
            self.game.ball.y,
            self.game.paddle1.x,
            self.game.paddle1.y
        ])
    
    def step(self,a,display=False):
        self.actions_taken.append(a)
        self.game.paddle1move(a,self.player_speed)
        self.game.computer_move(self.computer_speed)
        done = self.game.update()
        save_reward = 1 if self.check_save() else 0
        if done:
            if self.game.paddle1.score > self.game.paddle2.score:
                win_reward = 10
            else:
                win_reward = 0
            reward = win_reward + save_reward
        else:
            reward = save_reward
        self.__update_state_memory()
        s = self.get_state()
        
        return s, reward, done, {}
        
        
