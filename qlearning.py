import numpy as np
import Tiling
import random
import ast
from os import path
import pickle

np.seterr(all='raise')


class QTable:
    def __init__(self, _gamma, _tiles_per_dim, _lims, _tilings, _action_size, resume_last=False):

        self.name = 'Noone'
        self.alpha = 0.2
        self.gamma = _gamma
        self.epsilon = 0.05

        # s, a, r , the previous state, action, and reward, initially null
        self.previous_coded_state = np.NaN
        self.previous_Q_index = np.NaN
        self.previous_action = np.NaN
        self.previous_reward = np.NaN

        # Meta info
        self.Q_file = "Q_action_table.npy"
        self.Qn_file = "Qn_table.npy"
        self.Q_hash_file = "Q_hash.pickle"
        self.Q_hash_file_old = "Q_hash.txt"

        # Tile coder to discretize a continuous state space
        self.T = Tiling.TileCoder(_tiles_per_dim, _lims, _tilings)

        # Build arrays to store data
        self.initial_array_size = 1000000
        self.Q_hash = {}
        self.Q = np.array([np.zeros(_action_size)] * self.initial_array_size)
        self.Q_size = self.Q.size
        self.Qn = np.array([np.zeros(_action_size)] * self.initial_array_size)

        if resume_last:
            if self.files_exist():
                print("Loading previous agent")
                self.load_Q_table_from_file()
            else:
                print("Previous agent not found, creating new agent")

    def check_set_Q_hash(self, coded_state):
        if coded_state not in self.Q_hash:
            current_index = len(self.Q_hash)
            self.Q_hash[coded_state] = current_index + 1

    def get_Q_index(self, coded_state):
        # TODO: Handle if state doesnt exist in table
        return self.Q_hash[coded_state]

    def get_Q_actions(self, Q_index):
        return self.Q[Q_index]

    def set_Q(self, Q_index, action, new_Q):
        self.Q[Q_index][action] = new_Q

    def get_Q_Qn(self, Q_index, action):
        return self.Q[Q_index][action], self.Qn[Q_index][action]

    def increment_Qn(self, Q_index, action):
        self.Qn[Q_index][action] = self.Qn[Q_index][action] + 1

    def get_max_Q(self, Q_index):
        return np.amax(self.get_Q_actions(Q_index))

    def epsilon_greedy(self, Q_index):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.Q[Q_index])

    def save_Q_table_to_file(self, old_save=False):

        np.save(self.Q_file, self.Q)
        np.save(self.Qn_file, self.Qn)
        if old_save:
            with open(self.Q_hash_file_old, 'w') as fp:
                fp.write(str(self.Q_hash))
        else:
            with open(self.Q_hash_file, 'wb') as fp:
                pickle.dump(self.Q_hash, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_Q_table_from_file(self, old_save=False):
        self.Q = np.load(self.Q_file)
        self.Qn = np.load(self.Qn_file)
        if old_save:
            with open(self.Q_hash_file_old, 'r') as fp:
                loaded_string = fp.read()
                self.Q_hash = ast.literal_eval(loaded_string)
        else:
            with open(self.Q_hash_file, 'rb') as fp:
                self.Q_hash = pickle.load(fp)

    def files_exist(self):
        if path.exists(self.Q_file) and path.exists(self.Qn_file) and path.exists(self.Q_hash_file):
            return True
        else:
            return False

    def change_name(self, new_name):
        self.name = new_name

    def update_Q(self, current_state, current_reward, terminated):
        # Algorithm used from "Artificial Intelligence A Modern Approach" by Stuart Russell and Peter Norvig
        # Q-Learning-Agent, page 844, figure 21.8

        current_coded_state = self.T[current_state].tobytes()
        self.check_set_Q_hash(current_coded_state)

        current_Q_index = self.get_Q_index(current_coded_state)

        if terminated:               # Set as 0 because of the environment, 0 = no action
            self.set_Q(self.previous_Q_index, 0, current_reward)

        if ~(np.isnan(self.previous_Q_index)):
            self.increment_Qn(self.previous_Q_index, self.previous_action)
            previous_Q, Qn = self.get_Q_Qn(self.previous_Q_index, self.previous_action)
            this_Q = previous_Q + self.alpha * (
                np.clip(Qn * (self.previous_reward + self.gamma * self.get_max_Q(current_Q_index) - previous_Q), -1, 1))
            self.set_Q(self.previous_Q_index, self.previous_action, this_Q)

        self.previous_Q_index = current_Q_index
        self.previous_reward = current_reward
        self.previous_action = int(self.epsilon_greedy(current_Q_index))
        return self.previous_action
