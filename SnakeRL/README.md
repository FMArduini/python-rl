# SnakeRL

## Scripts:

- SnakeRL/train.py to train the algorithm. Models are saved in snakeRL/models
- SnakeRL/run_model.py to render the env with the latest saved model

## Environment

The environment is built as a gym environment. You can find it in gym_snake/

#### installation
install the gym-snake by navigating to the folder and using pip
>> cd gym_snake
>> pip install -e .

#### Screen
The environment consists of a numpy matrix representing the screen. Different values in this matrix indicate an object.
0 = empty
1 = wall
2 = snake body
3 = food

The environment is hard coded to be a 15x15 matrix, the snake always starts in the middle and is 3 units long, heading
upwards. These can be easily changed.

#### Actions
```
ACTION_SET = {
    0: np.array([-1, 0]),  # up
    1: np.array([1, 0]),  # down
    2: np.array([0, 1]),  # right
    3: np.array([0, -1])  # left
}
```

#### Rewards
Currently rewards are given by eating food. A small penalty is issued every step to avoid looping.

#### Rendering
Rendering this environment means simply printing the 'board' (numpy matrix) and clearing the screen after each step.
You can use the SnakeRL/test_snake_env.py script.


## Algorithm

The algorithm used will be from stable baselines.
Currently working with PPO2.

Every 10k iterations we stop and check performance.

## References
- https://github.com/hill-a/stable-baselines
- https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
