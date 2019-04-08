import sys, os

sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
print(sys.path)
from Neat import run_neat

config = os.getcwd() + '/Pong/config'
save_path = os.getcwd() + '/Pong/neat_saves'
run_neat(n_generations=1000, pop_size=1000, save_every=10, game_length=10,
         runs_per_net = 5, play_every=None, computer_paddle_speed=3,
         config_path=config, save_folder=save_path)
