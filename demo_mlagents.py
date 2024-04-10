from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.registry import default_registry
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import torch
import Rainbow_DQN
from gym.wrappers import FrameStack
import datetime

#unity_env = default_registry["WallJump"].make()
unity_env = UnityEnvironment(r"D:\Projects\Unity_builds\Walker_fast\UnityEnvironment.exe", no_graphics=True)
#env = UnityToGymWrapper(unity_env, uint8_visual=False, flatten_branched=True)
env = DummyVecEnv([lambda: UnityToGymWrapper(unity_env, uint8_visual=False, flatten_branched=True)])

print(torch.cuda.is_available())


try:
    agent = PPO("MlpPolicy", env, verbose=1)
    agent.learn(total_timesteps=20000000)
except KeyboardInterrupt:
    env.close()
agent.save(r"Agents\ppo_walker")
env.close()
start_time = datetime.datetime.now()
print("Starting")


unity_env_test = UnityEnvironment(r"D:\Projects\Unity_builds\Walker\UnityEnvironment.exe", no_graphics=False)
#env_test = UnityToGymWrapper(unity_env_test, uint8_visual=False, flatten_branched=True)
env_test = DummyVecEnv([lambda: UnityToGymWrapper(unity_env_test, uint8_visual=False, flatten_branched=True)])
observation = env_test.reset()
ack_reward = 0

try:
    while True:
        action = agent.predict(observation)  ### <-- This line samples a random action for from the environment. Replace this with your optimal action calculation ###
        observation, reward, terminated, info = env_test.step(action)
        ack_reward += reward

        if terminated:
            observation = env_test.reset()
            #print("reward: " + str(ack_reward))
            ack_reward = 0

except KeyboardInterrupt:
    print("Test ended")
    env_test.close()