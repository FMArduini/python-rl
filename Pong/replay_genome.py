from Neat import play_with_net
import neat
import pickle


def run_genome_file(file,config_path):
    config = pickle.load(open(config_path,'rb'))
    genome = pickle.load(open(file, 'rb'))
    print(genome.fitness)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    _ = play_with_net(net=net, display=True, computer_paddle_speed=3, game_length = 5)

def replay_game_file(file):
    game = pickle.load(open(file,'rb'))
    game.replay(canvas_name=file)
run_genome_file('Pong/neat_saves/best_gen_50.pkl',config_path = 'Pong/config/neat_saves/config')
#replay_game_file('Pong/neat_saves/game_file_gen70.pkl')

