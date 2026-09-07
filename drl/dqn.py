"""DQN: one fully separate Double-DQN learner (own network, target
network, optimizer, replay buffer) per sector -- no parameter sharing.
Simplest possible MARL baseline: 21 sectors learning completely
independently of each other, coordinated only implicitly through the
shared environment (a sector's reward already reflects the interference
its neighbors cause, via the overshoot term).
"""

import logging
import random

import numpy as np
import torch

from drl.base import TiltPolicy
from drl.replay_buffer import ReplayBuffer
from algorithm.mlp import Mlp
from algorithm.wesn import Wesn
from helpers.utils import get_logger

logger = get_logger(__name__)


class Dqn(TiltPolicy):
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
            spectral_radius, win_len, mlp_hidden_sizes, mlp_position) -- wesn only.
        :param num_features: int (same width for every sector) OR a
            [num_sectors] list of per-sector widths -- e.g. a sector with
            fewer than 4 neighbors getting a smaller state than one with
            all 4. Each sector already gets its own fully separate network
            (no parameter sharing), so a per-sector input width is a
            straightforward per-sector network-construction choice, not an
            architectural conflict.
        """
        super().__init__()

        # Initialization

        # I/O parameters
        self.num_sectors = num_sectors
        num_features_per_sector = (list(num_features) if isinstance(num_features, (list, tuple))
                                   else [num_features] * num_sectors)
        self.num_features = num_features_per_sector
        self.num_actions = num_actions

        # RL parameters
        self.target_update_steps = target_update_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.gamma = gamma

        # DL parameters
        self.model = model
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_steps_per_interval = train_steps_per_interval

        # Count of valid decision transitions completed per sector 
        self.sector_decision_intervals = np.zeros(num_sectors, dtype=np.int64)

        # Others
        self.device = device

        # One shared dedicated rng for all algorithm-side
        self.rng = random.Random(algorithm_seed)
        torch.manual_seed(algorithm_seed)

        # Network constructor for the selected model
        if model == "mlp":
            make_network = lambda nf: Mlp(nf, num_actions, list(hidden_sizes))
        elif model == "wesn":
            make_network = lambda nf: Wesn(nf, num_actions, **(wesn or {}))
        else:
            raise ValueError(f"Unknown model: {model!r}")

        # Policy networks
        self.networks = [make_network(num_features_per_sector[s]).to(device) for s in range(num_sectors)]

        # Target networks
        self.targets = [make_network(num_features_per_sector[s]).to(device) for s in range(num_sectors)]

        # Load models in policy and target networks
        for target, network in zip(self.targets, self.networks):
            target.load_state_dict(network.state_dict())

        # Optimizer
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=learning_rate)
                           for net in self.networks]

        # Replay buffer
        buffer_sequence_length = sequence_length if model == "wesn" else None
        self.buffers = [ReplayBuffer(replay_capacity, self.rng, sequence_length=buffer_sequence_length)
                        for _ in range(num_sectors)]

        num_features_repr = (str(num_features_per_sector[0]) if len(set(num_features_per_sector)) == 1
                             else f"variable {min(num_features_per_sector)}-{max(num_features_per_sector)}")
        logger.info("Dqn constructed: model=%s num_sectors=%d num_features=%s num_actions=%d "
                   "%s device=%s", model, num_sectors, num_features_repr, num_actions,
                   f"hidden_sizes={hidden_sizes}" if model == "mlp" else f"sequence_length={sequence_length}",
                   device)

    # Exploration and Exploitation
    def _epsilon(self, sector):
        epsilon = max(self.epsilon_end, self.epsilon_start * self.epsilon_decay_rate ** self.sector_decision_intervals[sector])
        logger.debug("Dqn._epsilon: exploration rate=%s", epsilon)

        return epsilon

    # Choose action
    def act(self, observations, training, mask=None, default_action=None):
        logger.function("Dqn.act start: training=%s", training)

        # Action initialization
        actions = (np.zeros(self.num_sectors, dtype=np.int64) if default_action is None
                  else np.array(default_action, dtype=np.int64).copy())

        # for each sector
        for sector, network in enumerate(self.networks):
            # Choose default action if observation is not meaningful
            if mask is not None and not mask[sector]:
                logger.debug("Dqn.act: No meaningful observation, choosing default action")
                continue

            # epsilon calculation for choosing exploration or exploitation
            epsilon = self._epsilon(sector) if training else 0.0

            # If exploration, choose random action
            if training and self.rng.random() < epsilon:
                actions[sector] = self.rng.randrange(self.num_actions)
                logger.debug("Dqn.act: choosing random action")
                continue

            # If exploitation, choose model based action
            if self.model == "wesn":
                buffer = self.buffers[sector]

                # Need sequence_length-1 past decisions buffered -- plus
                # the current observation, that's a full sequence_length
                # window. If not, choose random action.
                if len(buffer) < self.sequence_length - 1:
                    actions[sector] = self.rng.randrange(self.num_actions)
                    logger.debug("Dqn.act: buffer < sequence length, choosing random action")
                    continue

                # If enough data in buffer, choose model based action.
                # sequence_length=1 means no past history at all (a
                # length-1 window is just the current observation) --
                # skip recent() entirely rather than ask it for 0 items
                # from a possibly still-empty buffer, which has no
                # feature-width to fall back on if it's never stored
                # anything yet (see ReplayBuffer.recent's own n=0 handling
                # for what it CAN infer once the buffer is non-empty).
                n_history = self.sequence_length - 1
                if n_history == 0:
                    state_seq = observations[sector][None]
                else:
                    history = buffer.recent(n_history)
                    state_seq = np.concatenate([history, observations[sector][None]], axis=0)
                state = torch.as_tensor(state_seq, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                logger.debug("Dqn.act: choosing model based action")
            else:
                # Choose model based action
                state = torch.as_tensor(observations[sector], dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                logger.debug("Dqn.act: choosing model based action")

            # One hot encoding -> action index
            with torch.no_grad():
                actions[sector] = int(network(state).argmax(1).item())

        logger.function("Dqn.act end")
        return actions

    def observe(self, observations, actions, rewards, next_observations, terminal, mask=None):
        logger.function("Dqn.observe start: terminal=%s", terminal)
        num_trained = 0
        num_synced = 0

        for sector, buffer in enumerate(self.buffers):
            # Check if this observation is meaningful
            if mask is not None and not mask[sector]:
                continue  # not a real transition -- discard rather than learn from it

            # Update training step
            self.sector_decision_intervals[sector] += 1
            logger.debug("Dqn.observe: sector=%d decision_interval=%d",
                        sector, self.sector_decision_intervals[sector])

            # Update replay buffer
            buffer.add(observations[sector], actions[sector], rewards[sector],
                      next_observations[sector], float(terminal))
            logger.debug("Dqn.observe: Observation: %s, Action: %s, Reward: %s, Next Observation: %s",
                        observations[sector], actions[sector], rewards[sector], next_observations[sector])

            # Run training
            min_buffer = self.batch_size + (self.sequence_length - 1 if self.model == "wesn" else 0)
            if len(buffer) >= min_buffer:
                for _ in range(self.train_steps_per_interval):
                    self._learn(sector)
                num_trained += 1

            # Sync target Q network
            if self.sector_decision_intervals[sector] % self.target_update_steps == 0:
                self.targets[sector].load_state_dict(self.networks[sector].state_dict())
                num_synced += 1

        logger.debug("Dqn.observe: sectors_trained=%d sectors_synced=%d", num_trained, num_synced)
        logger.function("Dqn.observe end")

    def _learn(self, sector):
        logger.function("Dqn._learn start: sector=%d", sector)

        # Read transtion from the replay buffer
        obs, actions, rewards, next_obs, dones = self.buffers[sector].sample(self.batch_size)

        # wesn's obs/next_obs are [batch, sequence_length, features] windows;
        # only the window's LAST action/reward/done is the real training
        # target -- the earlier steps are just recurrent context.
        if self.model == "wesn":
            actions = actions[:, -1]
            rewards = rewards[:, -1]
            dones = dones[:, -1]
        logger.debug("_learn: sector=%d Observation: %s, Action: %s, Reward: %s, Next Observation: %s",
                    sector, obs, actions, rewards, next_obs)
        logger.debug("_learn: sector=%d obs shape=%s", sector, obs.shape)

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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("_learn: sector=%d chosen_q mean=%.4f target_q mean=%.4f",
                        sector, chosen_q.mean().item(), target_q.mean().item())

        # loss
        loss = torch.nn.functional.smooth_l1_loss(chosen_q, target_q)
        if not torch.isfinite(loss):
            logger.warning("_learn: sector=%d non-finite loss=%s", sector, loss.item())

        # Backpropagation
        self.optimizers[sector].zero_grad()
        loss.backward()
        self.optimizers[sector].step()
        logger.debug("_learn: sector=%d optimizer step done, loss=%.4f", sector, loss.item())

        # Update loss
        self.step_losses.append(loss.item())
        logger.function("Dqn._learn end: sector=%d loss=%.4f", sector, loss.item())

    def save(self, path):
        logger.function("Dqn.save start: path=%s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"networks": [net.state_dict() for net in self.networks],
                   "sector_decision_intervals": self.sector_decision_intervals}, path)
        logger.info("Dqn: saved %d sector networks to %s", self.num_sectors, path)
        logger.function("Dqn.save end")
