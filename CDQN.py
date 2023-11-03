import numpy as np
import random
from os import path
import pickle
from collections import deque
import itertools
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#device = "cpu"

np.seterr(all='raise')


class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class CnnQNetwork(nn.Module):
    def __init__(self, channels, pixel_hw, n_actions):
        super(CnnQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 5, stride=5)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=4)
        self.layer3 = nn.Linear(32*4*4, 256)
        self.layer4 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.layer3(x.reshape(x.size(0), -1)))
        return self.layer4(x)

class DQN():
    def __init__(self, _gamma, _alpha, _epsilon, _state_size, _action_size, CNN=False, resume_last=False):

        self.name = 'Noone'
        self.alpha = _alpha
        self.gamma = _gamma
        self.epsilon_end = _epsilon
        self.epsilon_start = 0.9
        self.epsilon_decay = 300
        self.state_size = _state_size
        self.action_size = _action_size
        self.buffer_size = 100000
        self.buff_index = 0
        self.CNN = CNN

        if(CNN):
            self.prediction_net = CnnQNetwork(3, 96, self.action_size).to(device)
            self.target_net = CnnQNetwork(3, 96, self.action_size).to(device) 
            if resume_last:
                if self.files_exist():
                    print("Loading previous agent")
                    self.prediction_net = torch.load("Agents/"+self.name)
                else:
                    print("Previous agent not found, creating new agent")
            self.target_net.load_state_dict(self.prediction_net.state_dict())
            self.optimizer = optim.RMSprop(self.prediction_net.parameters(), lr=self.alpha)
        else:
            self.prediction_net = QNetwork(self.state_size,  self.action_size).to(device)
            self.target_net = QNetwork(self.state_size,  self.action_size).to(device) 
            if resume_last:
                if self.files_exist():
                    print("Loading previous agent")
                    self.prediction_net = torch.load("Agents/"+self.name)
                else:
                    print("Previous agent not found, creating new agent")
            self.target_net.load_state_dict(self.prediction_net.state_dict())
            self.optimizer = optim.RMSprop(self.prediction_net.parameters(), lr=self.alpha)


        # Meta info
        self.model_name = "DQN_replay_table.pkl"
        self.replay_buffer_file_name = self.name+"_replay_buffer.pkl"
        # Build arrays to store data
        self.replay_buffer = deque([])

    def check_set_replay_transition(self, obs_prev, obs, action, reward, terminated):

        experience = [obs_prev]
        experience.append(obs)
        experience.append(action)
        experience.append(reward)
        experience.append(terminated)
        if self.buff_index >= self.buffer_size:
            self.replay_buffer.popleft()
        else:
            self.buff_index = self.buff_index + 1
        self.replay_buffer.append(experience)
            
    def split_experience(self, experience):
        return experience[0], experience[1], experience[2], experience[3], experience[4]       

    def save_agent_to_file(self):
        torch.save(self.prediction_net, "Agents/"+self.name) 

    def epsilon_greedy(self, state, steps_done):
        if(self.CNN):
            state = torch.from_numpy(state).float().to(device).unsqueeze(0).permute(0,3,1,2)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=device)
        random_sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * steps_done / self.epsilon_decay)
        if random_sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                prediction = self.prediction_net(state)
                return torch.argmax(prediction).item()
        else:
            return random.randint(0, self.action_size-1)

    def save_replay_buffer_to_file(self):
        with open(self.replay_buffer_file_name, 'wb') as fp:
            pickle.dump(self.replay_buffer, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_Q_table_from_file(self, old_save=False):
        with open(self.replay_buffer, 'rb') as fp:
            self.replay_buffer = pickle.load(fp)

    def files_exist(self):
        if path.exists(self.replay_buffer_file_name):
            return True
        else:
            return False

    def change_name(self, new_name):
        self.name = new_name

    def get_best_action(self, _current_state):
        current_state = torch.from_numpy(_current_state).float().to(device).unsqueeze(0).permute(0,3,1,2)
        #current_state = torch.tensor(_current_state, device=device, dtype=torch.float32)
        with torch.no_grad():
                prediction = self.prediction_net(current_state)
                return torch.argmax(prediction).item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.prediction_net.state_dict())
        return

    def get_replay_batch(self, batch_size):
        random_start = random.randint(0, self.buff_index)
        if random_start >= ((self.buff_index-1) - batch_size):
            random_start = ((self.buff_index-1) - batch_size)

        return deque(itertools.islice(self.replay_buffer, random_start, random_start+batch_size))

    def DQN_training(self, batch_size):
        if(self.buff_index < 10000):
            return

        # Algorithm used from "Implementing the Deep Q-Network", 2017

        # Get a sample batch from replay buffer
        batch = random.choices(self.replay_buffer, k=batch_size)
        #batch = self.get_replay_batch(batch_size=batch_size)
        T_batch = list(zip(*batch))
        #prev_obs_batch = torch.tensor(T_batch[0], device=device, dtype=torch.float32)
        prev_obs_batch = torch.from_numpy(np.array(T_batch[0])).float().to(device).permute(0,3,1,2)
        #obs_batch = torch.tensor(T_batch[1], device=device, dtype=torch.float32)
        obs_batch = torch.from_numpy(np.array(T_batch[1])).float().to(device).permute(0,3,1,2)
        action_batch = torch.tensor(T_batch[2], device=device, dtype=torch.float32)
        reward_batch = torch.tensor(T_batch[3], device=device, dtype=torch.float32)
        terminate_batch = torch.tensor(T_batch[4], device=device, dtype=torch.float32)
        terminate_batch = terminate_batch.long()
        terminate_batch = 1-terminate_batch

        target_values = self.target_net(obs_batch).max(1)[0]
        prediction_values = self.prediction_net(prev_obs_batch).gather(1, action_batch.reshape(batch_size, 1).long())

        criterion = nn.SmoothL1Loss()
        #loss = torch.mean(((reward_batch + self.gamma*terminate_batch*target_values) - prediction_values)**2)
        y = reward_batch + self.gamma*terminate_batch*target_values

        loss = criterion(prediction_values, y.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.prediction_net.parameters(), 1)
        self.optimizer.step()

        return