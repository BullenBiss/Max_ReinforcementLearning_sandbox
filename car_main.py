import gymnasium as gym
import CDQN
import Evaluation_tools
import torch
import numpy as np
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
env = gym.make("CarRacing-v2", continuous=False, render_mode=None)
env.action_space.seed(42)

observation, info = env.reset(seed=42)
action = env.action_space.sample()

img_h, img_w, img_c = env.observation_space.shape

gamma = 0.99
alpha = alpha = 1e-4
epsilon = 0.005
BATCH_SIZE = 32
agent = CDQN.DQN(gamma, alpha, epsilon, img_h, 5, CNN=True, resume_last=True)
evaluator = Evaluation_tools.Evaluator()

print("Starting")
terminated_i = 0
step_i = 0
i = 0
ack_reward = 0
plot_i = 0
try:
    while True:       
        i = i+1
        step_i = step_i+1
        observation_previous = observation
        action = agent.epsilon_greedy(observation, terminated_i)
        observation, reward, terminated, truncated, info = env.step(action)
        ack_reward = ack_reward + reward
        agent.check_set_replay_transition(observation_previous, observation, action, reward, terminated)
        
        if i >= 4:
            agent.DQN_training(BATCH_SIZE)
            i = 0

        if step_i >= 10:
            agent.update_target_network()
            step_i = 0

        if terminated or truncated:
            plot_i = plot_i + 1
            terminated_i = terminated_i + 1
            evaluator.cumulative_reward(ack_reward, terminated_i)
            print(ack_reward, terminated_i)
            ack_reward = 0

            if plot_i > 10:
                evaluator.plot_durations()
                plot_i = 0
            observation, info = env.reset()
except KeyboardInterrupt:
    evaluator.save_log()
    agent.save_agent_to_file()
    print("Run ended")
    env.close()

env_test = gym.make("CarRacing-v2",continuous=False, render_mode='human')
observation, info = env_test.reset(seed=42)
try:
    while True:
        action = agent.get_best_action(observation)
        observation, reward, terminated, truncated, info = env_test.step(action)

        if terminated or truncated:
            observation, info = env_test.reset()
except KeyboardInterrupt:
    print("Test ended")
    env_test.close()


