import gym
import gym_snake
env = gym.make('CartPole-v0')

from stable_baselines.common.policies import MlpPolicy
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
model = PPO2(MlpPolicy, env, verbose=1)
import os
modellist = os.listdir('models')
print(modellist)
latest = max([int(i.replace('ppo_model_','').replace('.zip','')) for i in modellist])
m = 'models/ppo_model_{}.zip'.format(latest)
model.load(load_path=m)
print('loaded',m)

done = False
state = env.reset()
while not done:
    env.render()
    action = model.predict(observation=state)[0]
    state, reward, done, info = env.step(action)
    print(action,reward,done)
    time.sleep(1)

