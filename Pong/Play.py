import pygame, sys, os

from pygame.locals import *
from Pong import Game


def play(game_length=None, game_time_limit=None, difficulty=3, fps = 60):
    if game_time_limit:
        seconds = game_time_limit//fps

    pygame.init()

    game = Game(game_length=game_length, game_time_limit=game_time_limit)

    # canvas declaration
    window = pygame.display.set_mode((game.globals.WIDTH, game.globals.HEIGHT), 0, 32)
    pygame.display.set_caption('PONG')

    clock = pygame.time.Clock()

    game_over = False
    counter = 0
    while game_over is False:
        counter += 1
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                game.keydown(event)
            elif event.type == KEYUP:
                game.keyup(event)
            elif event.type == QUIT:
                pygame.quit()
        game.computer_move(computer_speed=difficulty)
        game_over = game.update(log=True)
        game.display(window)
        game.display_seconds(window,seconds = counter // fps)
        pygame.display.update()
        clock.tick(fps)
    pygame.quit()
    return game


#game = play(game_time_limit=1000)