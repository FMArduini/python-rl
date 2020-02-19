import gym
import gym_snake
env = gym.make('snake-v0')

env.reset()
done=False
while not done:
    state,reward,done,_ = env.step(0)
    print(done)
    env.render()