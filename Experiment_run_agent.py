import gymnasium as gym
import Evaluation_tools
from gymnasium.wrappers import TransformReward
import torch
import Rainbow_DQN
from gym.wrappers import FrameStack
import datetime
from gymnasium.utils.play import PlayPlot, play
import numpy as np

### ============================================================ ###


### ============================================================ ###

print(torch.cuda.is_available())



#img_h = env.observation_space.shape
#gamma = 0.9
#alpha = 1e-3
gamma = 0.9
alpha = 2.5e-5

BATCH_SIZE = 128
#BATCH_SIZE = 256

agent = Rainbow_DQN.DQN(gamma, 
                alpha, 
                [4,84,84], 
                6,
                #4, 
                BATCH_SIZE,
                CNN=True, 
                resume_last=True,
                demonstration=False,
                agent_name="Experiment7_spaceInvaders")

#agent.change_name(experiment)

#env_test = gym.make("CarRacing-v2", continuous=False, render_mode=None)
#env_test = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
#env_test = mo.make("mo-supermario-v0", render_mode="human")
#env_test = gym.make("LunarLander-v2", render_mode="human")
#env_test._max_episode_steps = 100000
# Apply Wrappers to environment
env_test = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="human")
env_test = Rainbow_DQN.SkipFrame(env_test, skip=4)
env_test = Rainbow_DQN.GrayScaleObservation(env_test)
env_test = Rainbow_DQN.ResizeObservation(env_test, shape=84)
env_test = FrameStack(env_test, num_stack=4)

evaluator = Evaluation_tools.Evaluator(False)

observation, info = env_test.reset()
ack_reward = 0
step_i = 0
episode_i = 0
run_i = 0
episodes_data = evaluator.read_demonstrations(r"D:\Projects\Experiments\Experiment6_demonstration.txt")


if(False):
    try:
        for episode in episodes_data:
            for i in range(len(episode)):
                step_i = step_i+1
                episode_i = episode_i+1
                action = episode[i][1]  ### <-- This line samples a random action for from the environment. Replace this with your optimal action calculation ###
                observation, reward, terminated, truncated, info = env_test.step(action)
                ack_reward += reward
                #evaluator.store_for_log(step_i, episode_i, reward, ack_reward, action)

                if terminated or truncated:
                    observation, info = env_test.reset()
                    print("reward: " + str(ack_reward))
                    ack_reward = 0
                    episode_i = 0
                    run_i = run_i + 1
        env_test.close()
    except KeyboardInterrupt:
        print("Test ended")
        #env_test.close()

try:
    while run_i < 2:
        step_i = step_i+1
        episode_i = episode_i+1
        action = agent.select_action(observation)  ### <-- This line samples a random action for from the environment. Replace this with your optimal action calculation ###
        observation, reward, terminated, truncated, info = env_test.step(action)
        ack_reward += reward
        evaluator.store_for_log(step_i, episode_i, reward, ack_reward, action)

        if terminated or truncated:
            observation, info = env_test.reset()
            print("reward: " + str(ack_reward))
            ack_reward = 0
            episode_i = 0
            run_i = run_i + 1
    
    evaluator.save_log("Exp2_eval.txt")
    env_test.close()
except KeyboardInterrupt:
    print("Test ended")
    env_test.close()