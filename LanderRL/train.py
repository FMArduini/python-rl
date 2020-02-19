import gym
import gym_snake
import time
import numpy as np

env = gym.make('CartPole-v0')

from stable_baselines.common.policies import MlpPolicy,CnnPolicy
from stable_baselines import PPO2

import stable_baselines
print(stable_baselines)

import sys

from stable_baselines.common.env_checker import check_env
# It will check your custom environment and output additional warnings if needed


print('reset',env.reset(),env.reset().shape)
print(env.observation_space)

print('done')
# check_env(env)
# print('hey')

def run_model(model,env):
    s = env.reset()
    R = 0
    while True:
        a = model.predict(s)[0]

        s, r, d, _ = env.step(a)
        R += r
        if d:
            break
    return R

def run_many(n,model,env):
    return [run_model(model,env) for i in range(n)]


# Instantiate the env
#
# Define and Train the agent
# go to stable_baselines/common/policies.py in nature_cnn for the issue regarding cnn size (cannot subtract 4 from 3)
# for now we use a bigger image.

model = PPO2(MlpPolicy, env, verbose=0)
try:
    for i in range(10):
        model.learn(total_timesteps=1000)
        print("mean: {}".format(np.mean(run_many(10,model,env))))

except KeyboardInterrupt:
    print('Manual stop.')
finally:
    print("Saving model")
    model.save(save_path='models/ppo_model_{}'.format(int(time.time())))

