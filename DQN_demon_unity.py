import gymnasium as gym
import Evaluation_tools
from gymnasium.wrappers import TransformReward
import torch
import Rainbow_DQN
from gym.wrappers import FrameStack
import datetime
from gymnasium.utils.play import PlayPlot, play
import numpy as np
import pygame
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

### ============================================================ ###



def register_input(_keypress):
    global quit, restart
    keypress = _keypress
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                keypress = 4
            if event.key == pygame.K_RIGHT:
                keypress = 3
            if event.key == pygame.K_UP:
                keypress = 1
            if event.key == pygame.K_DOWN:
                keypress = 2
            if event.key == pygame.K_a:
                keypress = 5
            if event.key == pygame.K_d:
                keypress = 6
            if event.key == pygame.K_RETURN:
                restart = True
            if event.key == pygame.K_ESCAPE:
                keypress = -1       
        elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    keypress = 0  # Stop turning
                elif event.key == pygame.K_UP:
                    keypress = 0  # Stop accelerating
                elif event.key == pygame.K_DOWN:
                    keypress = 0   # Release brake
                elif event.key == pygame.K_a:
                    keypress = 0
                elif event.key == pygame.K_d:
                    keypress = 0
    return keypress

### ============================================================ ###

print(torch.cuda.is_available())


gamma = 0.90
alpha = 1e-5

BATCH_SIZE = 128
agent = Rainbow_DQN.DQN(gamma, 
                 alpha, 
                 243, 
                 39, 
                 BATCH_SIZE,
                 CNN=False, 
                 resume_last=False,
                 demonstration=False)
evaluator = Evaluation_tools.Evaluator()

agent.change_name("Unity")



start_time = datetime.datetime.now()
print("Starting")
terminated_i = 0
step_i = 0
i = 0
ack_reward = 0
plot_i = 0
ack_reward = 0
ack_action_reward = 0
ack1000_success = 0
ack1000_reward = 0
epsilon_travel = -0.1
demon_bool = agent.demonstration


if agent.demonstration:
    unity_env = UnityEnvironment(r"D:\Projects\Unity_builds\PushBlock_static\UnityEnvironment.exe", no_graphics=False)
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("CarRacing Controller")
    #env_demon = gym.make("LunarLander-v2", render_mode="human")
    env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=True)

    d_observation = agent.ConvertToTensor(env.reset())
    d_action = 0
    agent.demonstration_learning_rate(False)
    ack_reward = 0
    while True:
        d_observation_previous = d_observation
        d_action = register_input(d_action)
        if d_action == -1:
            break
        d_observation, d_reward, d_terminated, d_info = env.step(d_action)
        d_observation = agent.ConvertToTensor(d_observation)
        agent.check_set_replay_transition(d_observation_previous, d_observation, d_action, d_reward, d_terminated)
        ack_reward = ack_reward + d_reward
        if d_terminated:
            #evaluator.show_tensor(d_observation)
            print(ack_reward)
            d_observation = agent.ConvertToTensor(env.reset())
            agent.DQN_training()

    pygame.quit()
    env.close()

agent.update_target_network()

unity_env_fast = UnityEnvironment(r"D:\Projects\Unity_builds\Walker\UnityEnvironment.exe", no_graphics=False)
env_fast = UnityToGymWrapper(unity_env_fast, uint8_visual=True, flatten_branched=True)
observation = agent.ConvertToTensor(env_fast.reset())
#agent.demonstration_learning_rate(False)

action = 0
MAX_ITERATIONS = 10000
try:
    while terminated_i <= MAX_ITERATIONS:
        ### Main DQN segment ###
        i = i+1
        step_i = step_i+1

        observation_previous = observation
        action = agent.select_action(observation)

        observation, reward, terminated, info = env_fast.step(action)
        observation = agent.ConvertToTensor(observation)
        ack_reward = ack_reward + reward
        agent.check_set_replay_transition(observation_previous, observation, action, reward, terminated)
        
        # PER: increase beta
        fraction = min(terminated_i / MAX_ITERATIONS, 1.0)
        agent.per_beta = agent.per_beta + fraction * (1.0 - agent.per_beta)

        if i >= 4:
            agent.DQN_training()
            i = 0

        if step_i >= 5000:
            agent.update_target_network()
            step_i = 0
        ### ________________ ###

        ack_action_reward = 0

        ## === ##
        if terminated:         

            terminated_i = terminated_i + 1
            print(ack_reward, terminated_i)

            ack1000_reward += ack_reward
            evaluator.cumulative_reward(ack_reward, terminated_i)

            if(ack_reward >= 600):
                ack1000_success+=1

            if (terminated_i % 100 == 0):
                print("\niteration "+ str(terminated_i-100) +"-" + str(terminated_i) + ": " + str(ack1000_success/10) + "%" + " (mean reward: " + str(ack1000_reward/100) + ")")
                ack1000_success = 0
                ack1000_reward = 0

                evaluator.plot_durations()

            observation = agent.ConvertToTensor(env_fast.reset())
            ack_reward = 0

except KeyboardInterrupt:
    #Qtable.save_Q_table_to_file()
    print("Run ended")
    env_fast.close()

agent.save_agent_to_file()
evaluator.save_plots("Unity")
#evaluator.save_log("Unity")
stop_time = datetime.datetime.now()
print("Start time: ", start_time)
print("Stop time: ", stop_time)
### ============================================================ ###
'''
env_test = gym.make("CarRacing-v2", continuous=False, render_mode="human")
#env_test = gym.make("ALE/Breakout-v5", render_mode="human")
#env_test = mo.make("mo-supermario-v0", render_mode="human")
#env_test = gym.make("LunarLander-v2", render_mode="human")

# Apply Wrappers to environment

env_test = Rainbow_DQN.SkipFrame(env_test, skip=4)
env_test = Rainbow_DQN.GrayScaleObservation(env_test)
env_test = Rainbow_DQN.ResizeObservation(env_test, shape=84)
env_test = FrameStack(env_test, num_stack=4)
'''
unity_env = UnityEnvironment(r"D:\Projects\Unity_builds\Walker\UnityEnvironment.exe", no_graphics=False)
env_test = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=True)

observation= agent.ConvertToTensor(env_test.reset())
ack_reward = 0


try:
    while True:
        action = agent.select_action(observation)  ### <-- This line samples a random action for from the environment. Replace this with your optimal action calculation ###
        observation, reward, terminated, info = env_test.step(action)
        observation = agent.ConvertToTensor(observation)
        ack_reward += reward


        if terminated:
            observation = agent.ConvertToTensor(env_test.reset())
            
            print("reward: " + str(ack_reward))
            ack_reward = 0

except KeyboardInterrupt:
    print("Test ended")
    env_test.close() 