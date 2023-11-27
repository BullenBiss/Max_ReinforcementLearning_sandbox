import gymnasium as gym
import CDQN
import Evaluation_tools
import torch
import numpy as np
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
env_test = gym.make("CarRacing-v2",continuous=False, render_mode='human')
observation, info = env_test.reset(seed=42)
action = env_test.action_space.sample()

img_h, img_w, img_c = env_test.observation_space.shape

gamma = 0.99
alpha = alpha = 1e-4
epsilon = 0.005
BATCH_SIZE = 32
agent = CDQN.DQN(gamma, alpha, epsilon, img_h, 5, 'LongBoi', CNN=True, resume_last=True)
evaluator = Evaluation_tools.Evaluator()


try:
    while True:
        action = agent.get_best_action(observation)
        observation, reward, terminated, truncated, info = env_test.step(action)

        if terminated or truncated:
            observation, info = env_test.reset()
except KeyboardInterrupt:
    print("Test ended")
    env_test.close()


