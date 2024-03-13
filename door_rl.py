import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env(lambda: gym.make('AdroitHandDoor-v1', max_episode_steps=200))

model = PPO("MlpPolicy", vec_env, verbose=1, gamma=0.6, learning_rate=0.0004)
model.learn(total_timesteps=500000)
model.save("ppo_cartpole")

#del model # remove to demonstrate saving and loading

model.save("ppo_cartpole")

vec_env_test = make_vec_env(lambda: gym.make('AdroitHandDoor-v1', max_episode_steps=400, render_mode="human"), n_envs=1)
obs = vec_env_test.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env_test.step(action)


