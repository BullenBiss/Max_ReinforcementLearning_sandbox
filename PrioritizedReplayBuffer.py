from segment_tree import MinSegmentTree, SumSegmentTree
import numpy as np
import random
from typing import Dict, List, Tuple
import torch

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim, size: int, batch_size: int = 32, CNN=False):
        if CNN:
            self.obs_prev = np.empty((size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=np.float32)
            self.obs = np.empty((size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=np.float32)
        else: 
            self.obs_prev = np.empty((size, obs_dim), dtype=np.float32)
            self.obs = np.empty((size, obs_dim), dtype=np.float32)

        self.acts_buf = np.empty((size), dtype=np.float32)
        self.rews_buf = np.empty((size), dtype=np.float32)
        self.done_buf = np.empty((size), dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        self.obs_dim = obs_dim
        self.CNN = CNN
    def store(
        self,
        _obs_prev,
        act, 
        rew, 
        _obs, 
        done,
    ):

        if self.CNN:
            self.obs_prev[self.ptr] = torch.stack([x for x in _obs_prev]).cpu()
            self.obs[self.ptr] = torch.stack([x for x in _obs]).cpu()
        else:
            self.obs_prev[self.ptr] = torch.tensor(_obs_prev, dtype=torch.float32).cpu()
            self.obs[self.ptr] = torch.tensor(_obs, dtype=torch.float32).cpu()

        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return [self.obs_prev[idxs],
                self.obs[idxs],
                self.acts_buf[idxs],
                self.rews_buf[idxs],
                self.done_buf[idxs]]

    def __len__(self) -> int:
        return self.size



class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        obs_dim: int,
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        CNN = False
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size, CNN=CNN)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_prev[indices]
        next_obs = self.obs[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return[ 
            obs,
            next_obs,
            acts,
            rews,
            done,
            weights,
            indices,
        ]

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight