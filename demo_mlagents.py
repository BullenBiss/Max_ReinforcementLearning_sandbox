from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.registry import default_registry

import torch
import Rainbow_DQN
from gym.wrappers import FrameStack
import datetime

#unity_env = default_registry["WallJump"].make()
unity_env = UnityEnvironment("C:\\PROJECTS\\ml-agents\\Project\\example_envs\\UnityEnvironment.exe")
env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=True)


print(torch.cuda.is_available())

img_h= env.observation_space.shape
gamma = 0.9
alpha = alpha = 1e-5

BATCH_SIZE = 128
agent = Rainbow_DQN.DQN(gamma, 
                 alpha, 
                 img_h, 
                 7, 
                 BATCH_SIZE,
                 CNN=False, 
                 resume_last=False)

agent.change_name("Unity")

observation = env.reset()
action = env.action_space.sample()
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

MAX_ITERATIONS = 15000

try:
    while terminated_i <= MAX_ITERATIONS:
        ### Main DQN segment ###
        i = i+1
        step_i = step_i+1

        observation_previous = observation
        action = agent.select_action(observation, terminated_i)

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

            if(ack_reward >= 600):
                ack1000_success+=1

            if (terminated_i % 100 == 0):
                print("\niteration "+ str(terminated_i-100) +"-" + str(terminated_i) + ": " + str(ack1000_success/10) + "%" + " (mean reward: " + str(ack1000_reward/100) + ")")
                ack1000_success = 0
                ack1000_reward = 0
            
            observation = env.reset()
            ack_reward = 0

except KeyboardInterrupt:
    #Qtable.save_Q_table_to_file()
    print("Run ended")
    env.close()

agent.save_agent_to_file()
stop_time = datetime.datetime.now()
print("Start time: ", start_time)
print("Stop time: ", stop_time)
