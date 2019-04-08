import numpy as np
import neat
import pygame
from Pong import Game
import pickle
import shutil
import os
from collections import Counter

def play_with_net(net, display=True, log = False, computer_paddle_speed=3, game_length=10, canvas_name=None):
    actions = []
    game = Game(game_length=game_length)
    game_over = False

    if display:
        pygame.init()
        # canvas declaration
        window = pygame.display.set_mode((game.globals.WIDTH, game.globals.HEIGHT), 0, 32)
        pygame.display.set_caption('PONG' if not canvas_name else canvas_name)

        clock = pygame.time.Clock()

    while game_over is False:
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
        inpt = game.get_input()
        action = activate(net, inpt)
        actions.append(action)
        game.computer_move(computer_speed=computer_paddle_speed)
        game.neat_move(action, speed=computer_paddle_speed)
        game_over = game.update(log)
        if display:
            game.display(window)
            game.display_centre_text(window, 'fitness = {0:.4f}'.format(
                get_fitness(game, game_length = game_length, penalise_nomoves=True,penalise_fitness=False,penalise_oscillation=True)
            ))
            pygame.display.update()
            clock.tick(60)
    if display:
        pygame.quit()
    return game


def activate(net, input):
    output = net.activate(input)[0]
    return 1 if output > 0.5 else -1


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = play_with_net(net=net, display=False, computer_paddle_speed=config.computer_paddle_speed,
                         game_length=config.game_length)
    return get_fitness(game,
                       game_length = config.game_length,
                       penalise_fitness=config.penalise_fitness,
                       penalise_nomoves=config.penalise_nomoves,
                       penalise_oscillation=config.penalise_oscillations,
                       penalise_weight=config.penalise_weight)



def get_fitness(game, game_length, penalise_nomoves, penalise_oscillation, penalise_fitness, penalise_weight = 0.2):
    fitness = game.paddle1.score / game_length
    fitness += (game.paddle1.saves * 0.02)
    if penalise_nomoves:
        fitness = fitness * np.std(game.neat_actions)

    if penalise_oscillation:
        positions = [i['y'] for i in game.paddle1.position_log]
        hist_data = np.array(list(Counter(positions).values())) / len(positions)
        fitness = fitness * (1 -  np.max(hist_data/np.sum(hist_data)))

    return fitness


#
# def get_fitness(game, penalise_nomoves, penalise_oscillation, penalise_fitness, penalise_weight = 0.2):
#     fitness = game.paddle1.saves
#     if penalise_nomoves:
#         fitness = fitness * np.std(game.neat_actions)  # this std is at best 1 and worst 0.
#
#     if penalise_oscillation:
#         positions = [i['y'] for i in game.paddle1.position_log]
#         hist_data = np.array(list(Counter(positions).values())) / len(positions)
#         fitness = 0 if np.any(hist_data>0.4) else fitness
#
#     if penalise_fitness:
#         fitness = fitness - (game.paddle2.saves + 1)*penalise_weight
#
#     return fitness


def eval_genomes(genomes, config):
    # Play game and get results
    idx, genomes = zip(*genomes)

    # Calculate fitness and top score
    best_fitness = -9999999  # counter
    best_genome = genomes[0]  # set temporarily the first as one
    fitnesses = []  # all generation fitnesses
    for genome in genomes:
        genome_fitnesses = []
        for run in range(config.runs_per_net):
            fitness = eval_genome(genome, config)
            genome.fitness = fitness
            genome_fitnesses.append(fitness)
        mean_fitness = np.mean(genome_fitnesses)
        fitnesses.append(mean_fitness)
        if mean_fitness > best_fitness:
            best_genome = genome

    config.mean_gen_fitness.append(np.mean(fitnesses))

    if len(config.mean_gen_fitness) >= 5:
        if np.all(np.array(config.mean_gen_fitness[-5:]) >= (config.penalise_threshold)):
            config.penalise_fitness = True

    if config.play_every:
        if config.gen % config.play_every == 0:
            best_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
            _ = play_with_net(net=best_net, display=True, computer_paddle_speed=3, game_length=3,
                              canvas_name=' Generation : {}'.format(config.gen))

    if config.gen % config.save_every == 0:
        pickle.dump(best_genome, open(os.path.join(config.save_folder, 'best_gen_{}.pkl'.format(config.gen)), 'wb'))
        best_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        game = play_with_net(net=best_net,log=True,display=False)
        game.save_game(os.path.join(config.save_folder,'game_file_gen{}.pkl'.format(config.gen)))
    config.gen += 1

    print("Best score: {0:.2f}, Worst Score: {1:.2f}, Mean Score: {2:.2f}, median; {3:.2f}".format(
        np.max(fitnesses),
        np.min(fitnesses),
        np.mean(fitnesses),
        np.median(fitnesses),
    ))
    print('penalising with computer score = ', config.penalise_fitness)
    print('penalising no moves = ', config.penalise_nomoves)


def run_genome_file(file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config')
    net = neat.nn.FeedForwardNetwork.create(pickle.load(open(file, 'rb')), config)
    _ = play_with_net(net=net, display=True, computer_paddle_speed=3)


def check_runtime(config):
    if not os.path.isdir(config.save_folder):
        raise RuntimeError("save folder does not exist: {}".format(config.save_folder))


def run_neat(
        n_generations=10,
        pop_size=10,
        save_every=10,
        game_length=3,
        runs_per_net=1,
        play_every=None,
        computer_paddle_speed=3,
        config_path='your_path_to_config',
        save_folder='path_to_your_save_folder'
):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path, )

    # GA meta
    if pop_size:
        config.pop_size = pop_size
    config.n_generations = n_generations
    config.gen = 0

    # Operational
    config.save_folder = save_folder
    shutil.copy(config_path, os.path.join(config.save_folder, 'config'))
    config.runs_per_net = runs_per_net
    config.save_every = save_every
    config.play_every = play_every  # parameter controls how often to display a match in pygame None if never
    config.mean_gen_fitness = []
    config.penalise_fitness = False
    config.penalise_threshold = 20  # fitness to reach before penalisation occurs
    config.penalise_weight = 0.2
    config.penalise_nomoves = True
    config.penalise_oscillations = True


    # simulation properties
    config.game_length = game_length
    config.computer_paddle_speed = computer_paddle_speed

    check_runtime(config)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))

    # Run until we achieve n.
    winner = p.run(eval_genomes, n_generations)

    # dump
    pickle.dump(winner, open(os.path.join(config.save_folder, 'last_generation.pkl'), 'wb'))
