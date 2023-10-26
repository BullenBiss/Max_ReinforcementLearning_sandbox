import gymnasium as gym
import qlearning
from gymnasium.wrappers import TransformReward

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode=None)
env = TransformReward(env, lambda r: 1 if r == 1 else r-0.04)



gamma = 0.99
alpha = 0.01
epsilon = 0.2

Qtable = qlearning.QTable(gamma, alpha, epsilon, _action_size=4, _state_size= 65, resume_last=False)
Qtable.change_name("IsBörje")

observation, info = env.reset(seed=42)
action = env.action_space.sample()
print("Starting")
terminated_i = 0

### TRAINING LOOP
# Put your training algorithm here
try:
    while True:
        observation, reward, terminated, truncated, info = env.step(action)
        action = Qtable.update_Q(observation, reward, terminated)

        if terminated or truncated:
            terminated_i = terminated_i + 1

            if reward > 0:
                print("Found goal at iteration", terminated_i)

            observation, info = env.reset()
except KeyboardInterrupt:
    Qtable.save_Q_table_to_file()
    print("Run ended")
    env.close()


### TESTING LOOP
# Put your algorithm for taking the best action here
env_test = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode='human')
observation, info = env_test.reset(seed=42)
try:
    while True:
        observation, reward, terminated, truncated, info = env_test.step(action)
        action = Qtable.get_best_action(observation)

        if terminated or truncated:
            observation, info = env_test.reset()
except KeyboardInterrupt:
    print("Test ended")
    env_test.close()