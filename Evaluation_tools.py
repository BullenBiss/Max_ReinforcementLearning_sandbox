import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.animation as animation
import numpy as np
import torch
import multiprocessing


class Evaluator:
    def __init__(self):
        plt.ion()
        self.total_reward = []

    def plot_process(self):
        _plot_process = multiprocessing.Process(target=self.plot_durations)
        _plot_process.start()

    def cumulative_reward(self, current_reward, current_run):
        self.total_reward.append(current_reward)

    def save_log(self):
        array_to_file = np.array(self.total_reward)
        np.save('Reward_log', array_to_file)

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
        if len(durations_t) >= 50:
            means = durations_t.unfold(0, 50, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(49), means))
            plt.plot(means.numpy())

        plt.pause(0.01)  # pause a bit so that plots are updated
