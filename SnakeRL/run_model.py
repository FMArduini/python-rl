import gym
import gym_snake
env = gym.make('snake-v0')

from stable_baselines.common.policies import MlpPolicy,CnnPolicy
from stable_baselines import PPO2
import time
from stable_baselines.common.env_checker import check_env
# It will check your custom environment and output additional warnings if needed


print('reset',env.reset(),env.reset().shape)
print(env.observation_space)

print('done')
# check_env(env)
# print('hey')


# Instantiate the env
#
# Define and Train the agent
model = PPO2(CnnPolicy, env, verbose=1)
import os
modellist = os.listdir('models')
print(modellist)
latest = max([int(i.replace('ppo_model_','').replace('.zip','')) for i in modellist])
m = 'models/ppo_model_{}.zip'.format(latest)
model.load(load_path=m)
print('loaded',m)

done = False
state = env.reset()
R= 0
while not done:
    env.render()
    action = model.predict(observation=state)[0]
    state, reward, done, info = env.step(action)
    R += reward
    print(action,reward,done)
    time.sleep(0.7)

print('final reward', R)

