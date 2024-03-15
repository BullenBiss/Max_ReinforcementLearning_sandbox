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


unity_env = UnityEnvironment("C:\\PROJECTS\\ml-agents\\Project\\example_envs\\UnityEnvironment.exe")
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


#img_h = env.observation_space.shape
gamma = 0.5
alpha = 1e-2

BATCH_SIZE = 128
agent = Rainbow_DQN.DQN(gamma, 
                 alpha, 
                 #[img_h,img_w, img_c], 
                 210,
                 7, 
                 BATCH_SIZE,
                 CNN=False, 
                 resume_last=False,
                 demonstration=True)
evaluator = Evaluation_tools.Evaluator()

agent.change_name("Rainbow_LunarLander")

if agent.demonstration:
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("CarRacing Controller")


env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=True)

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

observation = env.reset()
action = 0
MAX_ITERATIONS = 10000
try:
    while terminated_i <= MAX_ITERATIONS:
        ### Main DQN segment ###
        i = i+1
        step_i = step_i+1

        observation_previous = observation
        
        if demon_bool:
            action = register_input(action)
            if action == -1:
                demon_bool = False
                action = 0
                pygame.quit()
        else:
            action = agent.select_action(observation)

        observation, reward, terminated, info = env.step(action)

        ack_reward = ack_reward + reward
        agent.check_set_replay_transition(observation_previous, observation, action, reward, terminated)
        
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

            observation = env.reset()
            ack_reward = 0

except KeyboardInterrupt:
    #Qtable.save_Q_table_to_file()
    print("Run ended")
    env.close()

agent.save_agent_to_file()
evaluator.save_plots("Rainbow_LunarLander")
evaluator.save_log("Rainbow_LunarLander")
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
    env_test.close() '''