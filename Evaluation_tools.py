import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.animation as animation
import numpy as np


class Evaluator:
    def __init__(self):
        plt.ion()
        self.total_reward = []

    def cumulative_reward(self, current_reward, current_run):
        self.total_reward.append(current_reward)

    def save_log(self):
        array_to_file = np.array(self.total_reward)
        np.save('Reward_log', array_to_file)