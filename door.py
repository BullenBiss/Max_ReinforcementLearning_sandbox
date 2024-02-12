import pprint
import minari
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data import huggingface_utils
from imitation.data.types import TrajectoryWithRew
from imitation.data.types import DictObs
from imitation.util.util import make_vec_env
from imitation.algorithms import density as db

SEED = 42
rng = np.random.default_rng(0)

dataset = minari.load_dataset('door-expert-v1', download=True)
ep_generator = dataset.iterate_episodes()

vec_env = DummyVecEnv([lambda: gym.make('AdroitHandDoor-v1', max_episode_steps=400)])
vec_env.reset()
trajectories = []
for episode in ep_generator:
    obs = episode.observations
    acts = np.array(episode.actions)
    infos = np.array(episode.infos)
    terminal = np.array(episode.terminations, dtype=bool)
    rews = np.array(episode.rewards)

    trajectory = TrajectoryWithRew(obs, acts, None, terminal, rews)
    trajectories.append(trajectory)
rollouts = np.array(trajectories)
#rollouts = huggingface_utils.trajectories_to_dataset(trajectories=trajectories)


#env = gym.make('AdroitHandDoor-v1', max_episode_steps=400)

imitation_trainer = PPO(
    ActorCriticPolicy, vec_env, learning_rate=3e-4, gamma=0.95, ent_coef=1e-4, n_steps=400, batch_size=10
)

density_trainer = db.DensityAlgorithm(
    venv=vec_env,
    rng=rng,
    demonstrations=rollouts,
    rl_algo=imitation_trainer,
    density_type=db.DensityType.STATE_ACTION_DENSITY,
    is_stationary=True,
    kernel="gaussian",
    kernel_bandwidth=0.4,
    standardise_inputs=True,
)
density_trainer.train()

def print_stats(density_trainer, n_trajectories):
    stats = density_trainer.test_policy(n_trajectories=n_trajectories)
    print("True reward function stats:")
    pprint.pprint(stats)
    stats_im = density_trainer.test_policy(true_reward=False, n_trajectories=n_trajectories)
    print("Imitation reward function stats:")
    pprint.pprint(stats_im)

print("Stats before training:")
print_stats(density_trainer, 1)

density_trainer.train_policy(10000)  # Train for 1_000_000 steps to approach expert performance.

print("Stats after training:")
print_stats(density_trainer, 1)




vec_env_test = DummyVecEnv([lambda: gym.make('AdroitHandDoor-v1', max_episode_steps=400, render_mode="human")])
obs = vec_env_test.reset()

while True:
    #action = agent.get_best_action(observation)
    action, _states = imitation_trainer.predict(obs)
    obs, rewards, dones, info = vec_env_test.step(action)
    vec_env_test.render("human")