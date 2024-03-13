import numpy as np
import random
from os import path
import pickle
from collections import deque
import itertools
import math
from gym.spaces import Box
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import PrioritizedReplayBuffer as PER
import noisy_layer
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#device = "cpu"

np.seterr(all='raise')


class QNetwork(nn.Module):
    def __init__(self, pixel_hw, n_actions):
        super(QNetwork, self).__init__()
        self.pixel_hw = pixel_hw
        self.noisy_layer1 = noisy_layer.NoisyLinear(8, 64)
        self.noisy_layer2 = noisy_layer.NoisyLinear(64, 512)
        self.noisy_layer3 = noisy_layer.NoisyLinear(512, n_actions)
        self.noisy_layer4 = noisy_layer.NoisyLinear(8, 64)
        self.noisy_layer5 = noisy_layer.NoisyLinear(64, 512)
        self.noisy_layer6 = noisy_layer.NoisyLinear(512, 1)

        self.advantage = nn.Sequential(
        self.noisy_layer1,
        nn.ReLU(),
        self.noisy_layer2,
        nn.ReLU(),
        self.noisy_layer3
        )

        self.value = nn.Sequential(
        self.noisy_layer4,
        nn.ReLU(),
        self.noisy_layer5,
        nn.ReLU(),
        self.noisy_layer6
        )

    def forward(self, x):
        value = self.value(x)
        advantage = self.advantage(x)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self):
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()
        self.noisy_layer3.reset_noise()
        self.noisy_layer4.reset_noise()
        self.noisy_layer5.reset_noise()
        self.noisy_layer6.reset_noise()
    
class CnnQNetwork(nn.Module):
    def __init__(self, channels, pixel_hw, n_actions):
        super(CnnQNetwork, self).__init__()
        self.channels = channels
        self.pixel_hw = pixel_hw
        self.noisy_layer1 = noisy_layer.NoisyLinear(3136, 512)
        self.noisy_layer2 = noisy_layer.NoisyLinear(512, n_actions)
        self.noisy_layer3 = noisy_layer.NoisyLinear(3136, 512)
        self.noisy_layer4 = noisy_layer.NoisyLinear(512, 1)

        self.conv = nn.Sequential(
        nn.Conv2d(channels, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten()
        )

        self.advantage = nn.Sequential(
        self.noisy_layer1,
        nn.ReLU(),
        self.noisy_layer2
        )

        self.value = nn.Sequential(
        self.noisy_layer3,
        nn.ReLU(),
        self.noisy_layer4
        )

    def forward(self, x):
        cnn = self.conv(x)
        value = self.value(cnn)
        advantage = self.advantage(cnn)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self):
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()
        self.noisy_layer3.reset_noise()
        self.noisy_layer4.reset_noise()


class DQN():
    def __init__(self, 
                 _gamma, 
                 _alpha, 
                 _state_size, 
                 _action_size, 
                 _batch_size,
                 agent_name='Noone', 
                 CNN=False, 
                 resume_last=False,
                 demonstration=False):

        self.batch_size = _batch_size
        self.name = agent_name
        self.alpha = _alpha
        self.gamma = _gamma
        self.state_size = _state_size
        self.action_size = _action_size
        self.buffer_size = 10000
        self.buff_index = 0
        self.CNN = CNN
        self.demonstration = demonstration
        ## Prioritized replay Buffer ##
        self.per_alpha = 0.2
        self.per_beta = 0.6
        self.prior_eps = 1e-6      

        if(CNN):
            self.prediction_net = CnnQNetwork(4, 84, self.action_size).to(device)
            self.target_net = CnnQNetwork(4, 84, self.action_size).to(device) 
            if resume_last:
                if self.files_exist():
                    print("Loading previous agent")
                    self.prediction_net = torch.load("Agents/"+self.name)
                else:
                    print("Previous agent not found, creating new agent")
            self.target_net.load_state_dict(self.prediction_net.state_dict())
            self.optimizer = optim.Adam(self.prediction_net.parameters(), lr=self.alpha)
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
            self.optimizer = optim.Adam(self.prediction_net.parameters(), lr=self.alpha)


        # Meta info
        self.model_name = "DQN_replay_table.pkl"
        self.replay_buffer_file_name = self.name+"_replay_buffer.pkl"
        # Build arrays to store data
        #self.replay_buffer = PER.ReplayBuffer(_state_size, self.buffer_size, self.batch_size)

        # PER
        self.memory = PER.PrioritizedReplayBuffer(
            _state_size, self.buffer_size, self.batch_size, _alpha, CNN
        )        

    def check_set_replay_transition(self, _obs_prev, _obs, action, reward, terminated):
        self.memory.store(_obs_prev, action, reward, _obs, terminated)
            
    def split_experience(self, experience):
        return experience[0], experience[1], experience[2], experience[3], experience[4]       

    def save_agent_to_file(self):
        torch.save(self.prediction_net, "Agents/"+self.name) 

    def select_action(self, _state):
        if self.CNN:
            state = torch.stack([x for x in _state]).unsqueeze(0)
        else: state = torch.tensor(_state, device=device, dtype=torch.float32)
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            prediction = self.prediction_net(state)
            return torch.argmax(prediction).item()

    def save_replay_buffer_to_file(self):
        with open(self.replay_buffer_file_name, 'wb') as fp:
            pickle.dump(self.replay_buffer, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_Q_table_from_file(self, old_save=False):
        with open(self.replay_buffer, 'rb') as fp:
            self.replay_buffer = pickle.load(fp)

    def files_exist(self):
        if path.exists("Agents/"+self.name):
            return True
        else:
            return False

    def change_name(self, new_name):
        self.name = new_name

    def get_best_action(self, _current_state):
        current_state = torch.stack([x for x in _current_state]).unsqueeze(0)
        with torch.no_grad():
                prediction = self.prediction_net(current_state)
                return torch.argmax(prediction).item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.prediction_net.state_dict())
        return
    
    def process_observation(self, observation, shape):
        transform = T.Grayscale()
        transforms = T.Compose(
            [T.Resize(shape, antialias=True), T.Normalize(0, 255)]
        )
    
        observation = np.transpose(observation, (2,0,1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        observation = transform(observation)
        observation = transforms(observation).squeeze(0)
        return observation
    

    def DQN_training(self):
        if(len(self.memory) < self.batch_size):
            return
        # Algorithm used from "Implementing the Deep Q-Network", 2017

        # Get a sample batch from replay buffer
        T_batch = self.memory.sample_batch(self.per_beta)
        
        if self.CNN:
            prev_obs_batch = torch.stack([x for x in torch.from_numpy(T_batch[0])]).cuda()
            obs_batch = torch.stack([x for x in torch.from_numpy(T_batch[1])]).cuda()
        else:
            prev_obs_batch = torch.tensor(T_batch[0], device=device, dtype=torch.float32).cuda()
            obs_batch = torch.tensor(T_batch[1], device=device, dtype=torch.float32).cuda()

        action_batch = torch.tensor(T_batch[2], device=device, dtype=torch.float32).reshape(-1, 1)
        reward_batch = torch.tensor(T_batch[3], device=device, dtype=torch.float32).reshape(-1, 1)
        terminate_batch = torch.tensor(T_batch[4], device=device, dtype=torch.float32).reshape(-1, 1)
        terminate_batch = terminate_batch.long()
        terminate_batch = 1-terminate_batch

    #PER weights and indices
        weights = torch.tensor(T_batch[5], device=device, dtype=torch.float32).reshape(-1, 1)
        indices = T_batch[6]

        #target_values = self.target_net(obs_batch).max(1)[0]
        target_values = self.target_net(obs_batch).gather(1, self.prediction_net(obs_batch).argmax(dim=1, keepdim=True)).detach()
        prediction_values = self.prediction_net(prev_obs_batch).gather(1, action_batch.reshape(self.batch_size, 1).long())

        #loss = torch.mean(((reward_batch + self.gamma*terminate_batch*target_values) - prediction_values)**2)
        y = reward_batch + self.gamma*terminate_batch*target_values
        y = y.to(device)

        elementwise_loss = F.smooth_l1_loss(prediction_values, y, reduction="none")        
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.prediction_net.parameters(), 100)
        self.optimizer.step()

        self.prediction_net.reset_noise()
        self.target_net.reset_noise()

    # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)       

        return
    
### =========================================== ###
    
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float, device=device)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation