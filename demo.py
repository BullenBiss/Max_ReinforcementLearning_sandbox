import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env



vec_env_test = make_vec_env(lambda: gym.make('AdroitHandDoor-v1', max_episode_steps=200, render_mode="human"), n_envs=1)
model1 = PPO.load("HandRL.zip", vec_env_test)
model2 = PPO.load("HandRL_long.zip", vec_env_test)
model3 = PPO.load("Hand2door.zip", vec_env_test)

obs = vec_env_test.reset()
try:
    while True:
        action, _states = model1.predict(obs)
        obs, rewards, dones, info = vec_env_test.step(action)
except KeyboardInterrupt:
    print("Test ended")

try:
    while True:
        action, _states = model2.predict(obs)
        obs, rewards, dones, info = vec_env_test.step(action)
except KeyboardInterrupt:
    print("Test ended")

try:
    while True:
        action, _states = model3.predict(obs)
        obs, rewards, dones, info = vec_env_test.step(action)
except KeyboardInterrupt:
    print("Test ended")
