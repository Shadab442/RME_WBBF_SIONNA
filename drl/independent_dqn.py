"""Independent DQN: one fully separate Double-DQN learner (own network,
target network, optimizer, replay buffer) per sector -- no parameter
sharing. Simplest possible MARL baseline: 21 sectors learning completely
independently of each other, coordinated only implicitly through the
shared environment (a sector's reward already reflects the interference
its neighbors cause, via the overshoot term).
"""

import random

import numpy as np
import torch

from drl.base import TiltPolicy
from drl.common import Mlp, ReplayBuffer


class IndependentDqn(TiltPolicy):
    def __init__(
        self,
        num_sectors: int,
        num_features: int,
        num_actions: int,
        hidden_sizes=(32,),
        learning_rate: float = 5e-4,
        gamma: float = 0.9,
        batch_size: int = 32,
        replay_capacity: int = 10_000,
        warmup_steps: int = 0,
        train_steps_per_interval: int = 1,
        target_update_steps: int = 50,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_rate: float = 0.985,
        algorithm_seed: int = 0,
        device: str = "cpu",
    ):
        """
        :param algorithm_seed: seeds EVERYTHING algorithm-side
        :param train_steps_per_interval: gradient steps taken per sector at
            each observe() call 
        """
        super().__init__()
        self.num_sectors = num_sectors
        self.num_features = num_features
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.train_steps_per_interval = train_steps_per_interval
        self.target_update_steps = target_update_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.device = device
        self.steps = 0

        # One shared dedicated rng for all algorithm-side 
        self.rng = random.Random(algorithm_seed)

        torch.manual_seed(algorithm_seed)
        self.networks = [Mlp(num_features, num_actions, list(hidden_sizes)).to(device)
                         for _ in range(num_sectors)]
        self.targets = [Mlp(num_features, num_actions, list(hidden_sizes)).to(device)
                        for _ in range(num_sectors)]
        for target, network in zip(self.targets, self.networks):
            target.load_state_dict(network.state_dict())
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=learning_rate)
                           for net in self.networks]
        self.buffers = [ReplayBuffer(replay_capacity, self.rng) for _ in range(num_sectors)]

    def _epsilon(self):
        return max(self.epsilon_end, self.epsilon_start * self.epsilon_decay_rate ** self.steps)

    def act(self, observations, training):
        epsilon = self._epsilon() if training else 0.0
        actions = np.zeros(self.num_sectors, dtype=np.int64)
        for sector, network in enumerate(self.networks):
            if training and self.rng.random() < epsilon:
                actions[sector] = self.rng.randrange(self.num_actions)
                continue
            with torch.no_grad():
                state = torch.as_tensor(observations[sector], dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                actions[sector] = int(network(state).argmax(1).item())
        return actions

    def observe(self, observations, actions, rewards, next_observations, terminal):
        self.steps += 1
        for sector, buffer in enumerate(self.buffers):
            buffer.add(observations[sector], actions[sector], rewards[sector],
                      next_observations[sector], float(terminal))
            if len(buffer) >= max(self.batch_size, self.warmup_steps):
                for _ in range(self.train_steps_per_interval):
                    self._learn(sector)
        if self.steps % self.target_update_steps == 0:
            for target, network in zip(self.targets, self.networks):
                target.load_state_dict(network.state_dict())

    def _learn(self, sector):
        obs, actions, rewards, next_obs, dones = self.buffers[sector].sample(self.batch_size)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).view(-1, 1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1)

        chosen_q = self.networks[sector](obs_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            # Double DQN: select the next action with the online network,
            # evaluate it with the target network.
            next_actions = self.networks[sector](next_t).argmax(1, keepdim=True)
            next_q = self.targets[sector](next_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * next_q
        loss = torch.nn.functional.smooth_l1_loss(chosen_q, target_q)
        self.optimizers[sector].zero_grad()
        loss.backward()
        self.optimizers[sector].step()
        self.step_losses.append(loss.item())

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"networks": [net.state_dict() for net in self.networks],
                   "steps": self.steps}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        for network, state in zip(self.networks, checkpoint["networks"]):
            network.load_state_dict(state)
        for target, network in zip(self.targets, self.networks):
            target.load_state_dict(network.state_dict())
        self.steps = checkpoint.get("steps", 0)
