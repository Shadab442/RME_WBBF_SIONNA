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
from algorithm.wesn import Wesn


class IndependentDqn(TiltPolicy):
    def __init__(
        self,
        num_sectors: int,
        num_features: int,
        num_actions: int,
        model: str = "wesn",
        hidden_sizes=(32,),
        sequence_length: int = 8,
        wesn=None,
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
        :param model: "mlp" | "wesn". MLP is a stateless per-sample Q-network
            (hidden_sizes only). WESN is a fixed random reservoir + one
            trained readout that reads a WINDOW of a sector's own last
            sequence_length decisions (see algorithm/wesn.py) -- gives the
            Q-network short-term memory across a sector's own history,
            which plain MLP lacks.
        :param sequence_length: wesn only -- window length (in this
            sector's own real decisions, not raw slots) fed through the
            reservoir each act()/learn() call.
        :param wesn: dict of Wesn kwargs (units, connectivity, leaky,
            spectral_radius, win_len) -- wesn only.
        """
        super().__init__()

        # Initialization
        self.num_sectors = num_sectors
        self.num_features = num_features
        self.num_actions = num_actions
        self.model = model
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.train_steps_per_interval = train_steps_per_interval
        self.target_update_steps = target_update_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.device = device
        self.sector_steps = np.zeros(num_sectors, dtype=np.int64)

        # One shared dedicated rng for all algorithm-side
        self.rng = random.Random(algorithm_seed)
        torch.manual_seed(algorithm_seed)

        # Network constructor for the selected model kind
        if model == "mlp":
            make_network = lambda: Mlp(num_features, num_actions, list(hidden_sizes))
        elif model == "wesn":
            make_network = lambda: Wesn(num_features, num_actions, **(wesn or {}))
        else:
            raise ValueError(f"Unknown model: {model!r}")

        # Policy networks
        self.networks = [make_network().to(device) for _ in range(num_sectors)]

        # Target networks
        self.targets = [make_network().to(device) for _ in range(num_sectors)]

        # Load models
        for target, network in zip(self.targets, self.networks):
            target.load_state_dict(network.state_dict())

        # Optimizer
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=learning_rate)
                           for net in self.networks]

        # Replay buffer -- windowed for wesn, plain i.i.d. transitions for mlp
        buffer_sequence_length = sequence_length if model == "wesn" else None
        self.buffers = [ReplayBuffer(replay_capacity, self.rng, sequence_length=buffer_sequence_length)
                        for _ in range(num_sectors)]

    def _epsilon(self, sector):
        return max(self.epsilon_end, self.epsilon_start * self.epsilon_decay_rate ** self.sector_steps[sector])

    def act(self, observations, training, mask=None, default_action=None):
        # Action initialization
        actions = (np.zeros(self.num_sectors, dtype=np.int64) if default_action is None
                  else np.array(default_action, dtype=np.int64).copy())

        # for each sector
        for sector, network in enumerate(self.networks):
            # Choose default action if observation is not meaningful
            if mask is not None and not mask[sector]:
                continue  

            # epsilon calculation
            epsilon = self._epsilon(sector) if training else 0.0

            # Choose random action
            if training and self.rng.random() < epsilon:
                actions[sector] = self.rng.randrange(self.num_actions)
                continue

            if self.model == "wesn":
                # Need sequence_length-1 real past decisions plus the
                # current one to form a window; not enough history yet ->
                # random (mirrors observe()'s own warmup gating below).
                buffer = self.buffers[sector]
                if len(buffer) < self.sequence_length - 1:
                    actions[sector] = self.rng.randrange(self.num_actions)
                    continue
                history = buffer.recent(self.sequence_length - 1)
                state_seq = np.concatenate([history, observations[sector][None]], axis=0)
                state = torch.as_tensor(state_seq, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
            else:
                # Choose model based action
                state = torch.as_tensor(observations[sector], dtype=torch.float32,
                                        device=self.device).unsqueeze(0)

            with torch.no_grad():
                actions[sector] = int(network(state).argmax(1).item())
        return actions

    def observe(self, observations, actions, rewards, next_observations, terminal, mask=None):
        for sector, buffer in enumerate(self.buffers):
            # Check if this observation is meaningful
            if mask is not None and not mask[sector]:
                continue  # not a real transition -- discard rather than learn from it

            # Update training step
            self.sector_steps[sector] += 1

            # Update replay buffer
            buffer.add(observations[sector], actions[sector], rewards[sector],
                      next_observations[sector], float(terminal))

            # Run training -- wesn also needs a full window's worth of history
            min_buffer = max(self.batch_size, self.warmup_steps,
                             self.sequence_length if self.model == "wesn" else 0)
            if len(buffer) >= min_buffer:
                for _ in range(self.train_steps_per_interval):
                    self._learn(sector)

            # Sync target Q network
            if self.sector_steps[sector] % self.target_update_steps == 0:
                self.targets[sector].load_state_dict(self.networks[sector].state_dict())

    def _learn(self, sector):
        # Read transtion from the replay buffer
        obs, actions, rewards, next_obs, dones = self.buffers[sector].sample(self.batch_size)

        # wesn's obs/next_obs are [batch, sequence_length, features] windows;
        # only the window's LAST action/reward/done is the real training
        # target -- the earlier steps are just recurrent context.
        if self.model == "wesn":
            actions = actions[:, -1]
            rewards = rewards[:, -1]
            dones = dones[:, -1]

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).view(-1, 1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).view(-1)

        # Q value
        chosen_q = self.networks[sector](obs_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            # Next action
            next_actions = self.networks[sector](next_t).argmax(1, keepdim=True)

            # Next Q value
            next_q = self.targets[sector](next_t).gather(1, next_actions).squeeze(1)

            # Target Q value
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        # loss
        loss = torch.nn.functional.smooth_l1_loss(chosen_q, target_q)

        # Backpropagation
        self.optimizers[sector].zero_grad()
        loss.backward()
        self.optimizers[sector].step()

        # Update loss
        self.step_losses.append(loss.item())

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"networks": [net.state_dict() for net in self.networks],
                   "sector_steps": self.sector_steps}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        for network, state in zip(self.networks, checkpoint["networks"]):
            network.load_state_dict(state)
        for target, network in zip(self.targets, self.networks):
            target.load_state_dict(network.state_dict())
        self.sector_steps = np.asarray(checkpoint.get("sector_steps", np.zeros(self.num_sectors, dtype=np.int64)))
