import gymnasium as gym
import DQN
import Evaluation_tools
import torch
import numpy as np
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
env = gym.make("LunarLander-v2", render_mode=None)
env.action_space.seed(42)
observation, info = env.reset(seed=42)
action = env.action_space.sample()

gamma = 0.99
alpha = alpha = 1e-4
epsilon = 0.05
BATCH_SIZE = 128
agent = DQN.DQN(gamma, alpha, epsilon, len(observation), 4, resume_last=True)
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
            ack_reward = 0
            if reward > -100:
                print(reward, terminated_i)
                #if reward >= 100:
                    #Qtable.save_Q_table_to_file()

            if plot_i > 10:
                evaluator.plot_durations()
                plot_i = 0
            observation, info = env.reset()
except KeyboardInterrupt:
    evaluator.save_log()
    agent.save_agent_to_file()
    print("Run ended")
    env.close()

env_test = gym.make("LunarLander-v2", render_mode='human')
observation, info = env_test.reset(seed=42)
try:
    while True:
        observation, reward, terminated, truncated, info = env_test.step(action)
        action = agent.get_best_action(observation)

        if terminated or truncated:
            observation, info = env_test.reset()
except KeyboardInterrupt:
    print("Test ended")
    env_test.close()


