import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.animation as animation
import numpy as np
import torch


class Evaluator:
    def __init__(self):
        plt.ion()
        plt.figure(figsize=(20,8))
        self.total_reward = []
        self.success = []
        self.mean_rewards = []
        self.iterations = []

    def cumulative_reward(self, current_reward, current_run):
        self.total_reward.append(current_reward)

    def save_log(self, name):
        #array_to_file = np.array(self.total_reward)
        #np.save('Reward_log', array_to_file)
        mat = [self.iterations, self.mean_rewards, self.success]
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