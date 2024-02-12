import minari
import gymnasium as gym
from stable_baselines3 import PPO

#dataset = minari.load_dataset('door-expert-v1', download=True)
env = gym.make('AdroitHandDoor-v1', max_episode_steps=400)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
obs = env.reset()
model.save("ppo_door")

while True:
    #action = agent.get_best_action(observation)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")

