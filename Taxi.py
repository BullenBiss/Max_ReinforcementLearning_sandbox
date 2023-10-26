import gymnasium as gym
import qlearning
from gymnasium.wrappers import TransformReward

env = gym.make("CarRacing-v2", domain_randomize=True, continuous=False, render_mode=None)

gamma = 0.7
alpha = 0.001
epsilon = 0.2

Qtable = qlearning.QTable(gamma, alpha, epsilon, _action_size=5, _state_size= 9216, resume_last=False)
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
        print(observation)
        action = Qtable.update_Q(observation, reward, terminated)

        if terminated or truncated:
            terminated_i = terminated_i + 1

            if reward > 700:
                print("Found goal at iteration", terminated_i, reward)

            observation, info = env.reset()
except KeyboardInterrupt:
    Qtable.save_Q_table_to_file()
    print("Run ended")
    env.close()


### TESTING LOOP
# Put your algorithm for taking the best action here
env_test = gym.make("CarRacing-v2", domain_randomize=True, continuous=False, render_mode='human')
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