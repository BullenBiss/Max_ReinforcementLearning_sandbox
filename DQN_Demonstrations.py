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
import keyboard
import time
### ============================================================ ###

def progress_preview(_evaluator, iteration, _agent):
    #env_preview = gym.make("CarRacing-v2", continuous=False, render_mode=None)
    env_preview = gym.make("ALE/Breakout-v5")
    #env_preview = mo.make("mo-supermario-v0")
    #env_preview = mo.LinearReward(env_preview)
    #env_preview = gym.make("LunarLander-v2", render_mode=None)
    #env_preview = TransformReward(env_preview, lambda r: 1 if r == 1 else r-0.04)

    # Apply Wrappers to environment
    env_preview = Rainbow_DQN.SkipFrame(env_preview, skip=4)
    env_preview = Rainbow_DQN.GrayScaleObservation(env_preview)
    env_preview = Rainbow_DQN.ResizeObservation(env_preview, shape=84)
    env_preview = FrameStack(env_preview, num_stack=4)

    p_observation, info = env_preview.reset()
    p_ack_reward = 0
    p_ack100_success = 0
    p_ack100_reward = 0
    p_action = 0
    p_terminated_i = 0

    while True:
        p_action = _agent.select_action(p_observation)
        p_observation, p_reward, p_terminated, p_truncated, p_info = env_preview.step(p_action)
        p_ack_reward += p_reward
        p_ack100_reward += p_reward


        if p_terminated or p_truncated:
            p_observation, p_info = env_preview.reset()
            p_terminated_i += 1

            if p_ack_reward >= 200:
                p_ack100_success += 1
            p_ack_reward = 0
            
            if p_terminated_i >= 10:
                print("preview result: " + str(p_ack100_success) + "% (mean reward: " + str(p_ack100_reward/10) + ")",end="\n", flush=True)
                _evaluator.save_rewards(p_ack100_success, p_ack100_reward/10, iteration)
                break

    env_preview.close()

def register_input(_keypress):
    global quit, restart
    keypress = _keypress
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                keypress = 2
            if event.key == pygame.K_RIGHT:
                keypress = 3
            if event.key == pygame.K_UP:
                keypress = 1
            if event.key == pygame.K_DOWN:
                keypress = 4
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
    return keypress

### ============================================================ ###

print(torch.cuda.is_available())


#img_h = env.observation_space.shape
gamma = 0.9
alpha = 1e-3
#gamma = 0.9
#alpha = 2.5e-5

#BATCH_SIZE = 128
BATCH_SIZE = 128

agent = Rainbow_DQN.DQN(gamma, 
                 alpha, 
                 [4,84,84], 
                 #8,
                 5, 
                 BATCH_SIZE,
                 CNN=True, 
                 resume_last=False,
                 demonstration=True)

agent.change_name("Experiment1_racecar_d")



if agent.demonstration:
    evaluator_demonstrations = Evaluation_tools.Evaluator()
    time_step = 0
    ep_time_step = 0
    d_ack_reward = 0
    #env_demon = gym.make("BreakoutNoFrameskip-v4",  render_mode="human")
    #env_demon = mo.make("mo-supermario-v0", render_mode="human")
    #env_demon = mo.LinearReward(env_demon)
    #env_demon = gym.make("LunarLander-v2", render_mode="human")
    #env_demon = TransformReward(env_demon, lambda r: 1 if r == 1 else r-0.04)
    env_demon = gym.make("CarRacing-v2", continuous=False, render_mode="human")
    env_demon = Rainbow_DQN.SkipFrame(env_demon, skip=4)
    env_demon = Rainbow_DQN.GrayScaleObservation(env_demon)
    env_demon = Rainbow_DQN.ResizeObservation(env_demon, shape=84)
    env_demon = FrameStack(env_demon, num_stack=4)

    d_observation, d_info = env_demon.reset()
    d_action = 0
    agent.demonstration_learning_rate(False)
    while True:
        if keyboard.is_pressed('w'):  # Move up or forward
            d_action = 3
        elif keyboard.is_pressed('a'):  # Move left
            d_action = 2
        elif keyboard.is_pressed('s'):  # Move down or backward
            d_action = 4
        elif keyboard.is_pressed('d'):  # Move right
            d_action = 1
        elif keyboard.is_pressed("enter"):  # Special action or select
            break
        else: d_action = 0

        time_step = time_step + 1
        ep_time_step = ep_time_step + 1

        d_observation_previous = d_observation
        d_observation, d_reward, d_terminated, d_truncated, d_info = env_demon.step(d_action)
        d_ack_reward = d_ack_reward + d_reward
        evaluator_demonstrations.store_for_log(time_step, ep_time_step, d_reward, d_ack_reward, d_action)
        #agent.check_set_replay_transition(d_observation_previous, d_observation, d_action, d_reward, d_terminated)
        #time.sleep(0.1)
        if d_truncated or d_terminated:
            print(time_step)
            ep_time_step = 0
            d_observation, d_info = env_demon.reset()
            agent.DQN_training()
    env_demon.close()
    evaluator_demonstrations.save_log("Experiment1_demonstration")
    #agent.update_target_network()

evaluator = Evaluation_tools.Evaluator()

#env = gym.make("BreakoutNoFrameskip-v4")
#env._max_episode_steps = 10000
#env = mo.make("mo-supermario-v0")
#env = mo.LinearReward(env)
env = gym.make("CarRacing-v2", continuous=False, render_mode=None)
#env = gym.make("LunarLander-v2", render_mode=None)
#env = TransformReward(env, lambda r: 1 if r == 1 else r-0.04)

# Apply Wrappers to environment
env = Rainbow_DQN.SkipFrame(env, skip=4)
env = Rainbow_DQN.GrayScaleObservation(env)
env = Rainbow_DQN.ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

observation, info = env.reset()
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
agent.demonstration_learning_rate(False)

total_timesteps = 0
episode_timesteps = 0
MAX_ITERATIONS = 1000
try:
    while terminated_i <= MAX_ITERATIONS:
        ### Main DQN segment ###
        i = i+1
        step_i = step_i+1
        total_timesteps = total_timesteps + 1
        episode_timesteps = episode_timesteps + 1

        observation_previous = observation
        action = agent.select_action(observation)

        observation, reward, terminated, truncated, info = env.step(action)

        ack_reward = ack_reward + reward

        agent.check_set_replay_transition(observation_previous, observation, action, reward, terminated)
        evaluator.store_for_log(total_timesteps, episode_timesteps, reward, ack_reward, action)
        # PER: increase beta
        fraction = min(terminated_i / MAX_ITERATIONS, 1.0)
        agent.per_beta = agent.per_beta + fraction * (1.0 - agent.per_beta)

        if i >= 4:
            agent.DQN_training()
            i = 0

        if step_i >= 10:
            agent.update_target_network()
            step_i = 0
        ### ________________ ###

        ack_action_reward = 0

        ## === ##
        if terminated or truncated:         
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
                #progress_preview(evaluator, terminated_i, agent)
            
            observation, info = env.reset()
            ack_reward = 0

except KeyboardInterrupt:
    #Qtable.save_Q_table_to_file()
    print("Run ended")
    env.close()

agent.save_agent_to_file()
evaluator.save_plots("Experiment1_racecar_d")
evaluator.save_log("Experiment1_racecar_d")
stop_time = datetime.datetime.now()
print("Start time: ", start_time)
print("Stop time: ", stop_time)
### ============================================================ ###

env_test = gym.make("CarRacing-v2", continuous=False, render_mode="human")
#env_test = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
#env_test = mo.make("mo-supermario-v0", render_mode="human")
#env_test = gym.make("LunarLander-v2", render_mode="human")
#env_test._max_episode_steps = 100000
# Apply Wrappers to environment

env_test = Rainbow_DQN.SkipFrame(env_test, skip=4)
env_test = Rainbow_DQN.GrayScaleObservation(env_test)
env_test = Rainbow_DQN.ResizeObservation(env_test, shape=84)
env_test = FrameStack(env_test, num_stack=4)

observation, info = env_test.reset()
ack_reward = 0


try:
    while True:
        action = agent.select_action(observation)  ### <-- This line samples a random action for from the environment. Replace this with your optimal action calculation ###
        observation, reward, terminated, truncated, info = env_test.step(action)

        ack_reward += reward


        if terminated or truncated:
            observation, info = env_test.reset()
            #print("reward: " + str(ack_reward))
            ack_reward = 0

except KeyboardInterrupt:
    print("Test ended")
    env_test.close()