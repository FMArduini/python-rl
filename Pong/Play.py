import pygame, sys, os


from pygame.locals import *
from Pong import Game

def play(game_length=5, difficulty=3):
    pygame.init()

    game = Game(game_length=game_length)

    # canvas declaration
    window = pygame.display.set_mode((game.globals.WIDTH, game.globals.HEIGHT), 0, 32)
    pygame.display.set_caption('PONG')

    clock = pygame.time.Clock()

    game_over = False
    while game_over is False:
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
        pygame.display.update()
        clock.tick(60)
    pygame.quit()
    return game

game = play()

game.replay()