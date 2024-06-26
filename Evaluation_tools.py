import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.animation as animation
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd

class Evaluator:
    def __init__(self, create_plot=True):
        if create_plot:
            plt.ion()
            plt.figure(figsize=(20,8))
        self.total_reward = []
        self.success = []
        self.mean_rewards = []

        self.total_timesteps = []
        self.episode_timesteps = []
        self.iteration_reward = []
        self.episode_reward = []
        self.action = []
        self.demonstration = None

    def cumulative_reward(self, current_reward, current_run):
        self.total_reward.append(current_reward)

    def store_for_log(self, total_timesteps, episode_timesteps, reward, ack_reward, action):
        self.total_timesteps.append(total_timesteps)
        self.episode_timesteps.append(episode_timesteps)
        self.iteration_reward.append(reward)
        self.episode_reward.append(ack_reward)
        self.action.append(action)

    def save_log(self, name):
        #array_to_file = np.array(self.total_reward)
        #np.save('Reward_log', array_to_file)
        mat = [self.total_timesteps, self.episode_timesteps, self.iteration_reward, self.episode_reward, self.action]
        transposed = list(map(lambda *x: list(x), *mat))

        with open(name+".txt", 'w') as file:
            for row in transposed:
                s = " ".join(map(str, row))
                file.write(s+'\n')

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.total_reward, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.01)  # pause a bit so that plots are updated

    def save_plots(self, name):
        plt.savefig(name+'.png')

    def save_rewards(self, success, mean_reward, iteration):
        self.success.append(success)
        self.mean_rewards.append(mean_reward)
        self.iterations.append(iteration)

    def show_tensor(self, tensor):
        transform = T.ToPILImage()
        img = transform(tensor)
        img.show()

    def read_demonstrations(self, filename):
        # Read the specific columns from the file using pandas
        data = pd.read_csv(filename, header=None, usecols=[1, 4], names=['EpisodeTimeStep', 'Action'], delim_whitespace=True)
        
        # Group data by the resetting of EpisodeTimeStep
        # Identify new episodes by checking where the timestep resets to a lower value than the previous one
        data['NewEpisode'] = data['EpisodeTimeStep'].diff().fillna(1) < 0
        data['EpisodeIndex'] = data['NewEpisode'].cumsum() - 1
        
        # Group by EpisodeIndex and create separate numpy arrays for each episode
        grouped = data.groupby('EpisodeIndex')
        episodes = [group[['EpisodeTimeStep', 'Action']].values for name, group in grouped]
        return np.array(episodes, dtype=object)

