import pprint
import minari
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.types import TrajectoryWithRew
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.data import huggingface_utils

SEED = 42
rng = np.random.default_rng(0)

dataset = minari.load_dataset('door-expert-v1', download=True)
ep_generator = dataset.iterate_episodes()

env = DummyVecEnv([lambda: gym.make('AdroitHandDoor-v1', max_episode_steps=400)])
env.reset()
trajectories = []
for episode in ep_generator:
    obs = episode.observations
    acts = np.array(episode.actions)
    infos = np.array(episode.infos)
    terminal = np.array(episode.terminations, dtype=bool)
    rews = np.array(episode.rewards)

    trajectory = TrajectoryWithRew(obs, acts, None, terminal, rews)
    trajectories.append(trajectory)
#rollouts = np.array(trajectories)
rollouts = huggingface_utils.trajectories_to_dataset(trajectories=trajectories)
#rollouts = huggingface_utils.TrajectoryDatasetSequence(demonstrations)
#env = gym.make('AdroitHandDoor-v1', max_episode_steps=400)

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=SEED,
)
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=201,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

env.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True
)

gail_trainer.train(200_000)


vec_env_test = DummyVecEnv([lambda: gym.make('AdroitHandDoor-v1', max_episode_steps=400, render_mode="human")])
obs = vec_env_test.reset()

while True:
    #action = agent.get_best_action(observation)
    action, _states = sqil_trainer.predict(obs)
    obs, rewards, dones, info = vec_env_test.step(action)
    vec_env_test.render("human")