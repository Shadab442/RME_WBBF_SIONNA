"""PPO: one fully separate continuous-action PPO learner (own actor,
critic, optimizer, on-policy rollout buffer) per sector -- no parameter
sharing, matching Dqn's own MARL convention (see drl/dqn.py's module
docstring). The only structural difference from Dqn: PPO is on-policy, so
it keeps a small per-sector rollout buffer (plain list, cleared after every
update) instead of Dqn's off-policy ReplayBuffer, and updates in a batch
every `rollout_length` decisions rather than one gradient step per observe()
call.

Action representation: TiltPolicy.act() still returns one INT tilt index
per sector, exactly like Dqn/RandomPolicy -- but here the index comes from
floor-quantizing a continuous sample from a per-sector Gaussian actor onto
`action_grid_deg` (typically a much finer grid than Dqn's own
downtilt_sweep_deg -- see SimulationEngine's drl_action_grid_deg), not from
an argmax over a small discrete action head. The underlying Gaussian policy
is trained on the RAW pre-quantization sample (log_prob evaluated there);
quantization is treated as a fixed post-processing step outside the
differentiable sampling path, the standard/simplest treatment for a
continuous policy whose action is discretized before being applied.

model="wesn" (this codebase's default for BOTH Dqn and Ppo): actor/critic
are Wesn networks (algorithm/gradient_based_wesn.py) reading a window of
this sector's own last sequence_length real observations, exactly like
Dqn's own wesn model -- see _build_state. A small per-sector ReplayBuffer
(history_buffers) supplies that window via its .recent() method ONLY;
unlike Dqn's own buffers, these are never sampled from (PPO is on-policy,
trained from self.rollouts below, not off-policy replay) -- reused purely
because it already implements the exact windowing Dqn's own _build_state
relies on, not for its replay-sampling capability.
"""

import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from drl.base import TiltPolicy
from drl.replay_buffer import ReplayBuffer
from algorithm.mlp import Mlp
from algorithm.gradient_based_wesn import Wesn
from helpers.utils import get_logger

logger = get_logger(__name__)


class GaussianActor(nn.Module):
    """Mean from an injected backbone network (Mlp or Wesn, output_size=1);
    log_std is a separate learned scalar parameter (state-independent) --
    the common simple-PPO default, not a second network output."""

    def __init__(self, mean_net: nn.Module, init_log_std: float):
        super().__init__()
        self.mean_net = mean_net
        self.log_std = nn.Parameter(torch.full((1,), float(init_log_std)))

    def forward(self, inputs):
        mean = self.mean_net(inputs).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class Ppo(TiltPolicy):
    def __init__(
        self,
        num_sectors: int,
        num_features,
        action_grid_deg,
        model: str = "wesn",
        hidden_sizes=(64,),
        sequence_length: int = 16,
        wesn=None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        rollout_length: int = 32,
        epochs_per_update: int = 4,
        minibatch_size: int = 32,
        init_log_std: float = 0.5,
        algorithm_seed: int = 0,
        device: str = "cpu",
        expose_diagnostics: bool = False,
    ):
        """
        :param action_grid_deg: [num_actions] ascending, evenly-spaced
            candidate tilts [deg] -- act()'s raw continuous sample is
            clamped to [action_grid_deg[0], action_grid_deg[-1]] then
            floor-quantized to this grid's own spacing; the returned index
            is this grid's index, NOT necessarily the same grid Dqn/
            AdaptiveLegacy use (see SimulationEngine.drl_action_grid_deg).
        :param model: "mlp" | "wesn" -- same two options and same meaning
            as Dqn's own model param (see drl/dqn.py): mlp is a stateless
            per-sample backbone (hidden_sizes only); wesn reads a WINDOW of
            this sector's own last sequence_length real observations (see
            _build_state), giving the actor/critic short-term memory across
            a sector's own decision history, which plain mlp lacks.
        :param sequence_length: wesn only -- window length (in this
            sector's own real decisions) fed through the reservoir each
            act()/learn() call -- same meaning as Dqn's own param.
        :param wesn: dict of Wesn kwargs (units, connectivity, leaky,
            spectral_radius, win_len, mlp_hidden_sizes, mlp_position) --
            wesn only, same as Dqn's own param.
        :param rollout_length: decisions collected per sector before that
            sector's own PPO update runs (and its buffer clears) --
            on-policy, unlike Dqn's per-observe() training from a replay
            buffer.
        :param epochs_per_update, minibatch_size: standard clipped-
            surrogate PPO update over the just-collected rollout.
        :param init_log_std: starting value for each sector's own learned
            (state-independent) log_std parameter.
        :param expose_diagnostics: accepted for factory-signature parity
            with Dqn; PPO doesn't currently populate last_decision_info
            (no q_values-shaped concept to expose) -- diagnostics logging
            degrades gracefully (see DrlDiagnosticsLogger.log_decision's
            own decision_info=None handling).
        """
        super().__init__()

        self.num_sectors = num_sectors
        num_features_per_sector = (list(num_features) if isinstance(num_features, (list, tuple))
                                   else [num_features] * num_sectors)
        self.num_features = num_features_per_sector

        self.action_grid_deg = np.asarray(action_grid_deg, dtype=float)
        self.num_actions = len(self.action_grid_deg)
        self.theta_min = float(self.action_grid_deg[0])
        self.theta_max = float(self.action_grid_deg[-1])
        self.step = float(self.action_grid_deg[1] - self.action_grid_deg[0]) if self.num_actions > 1 else 1.0

        self.model = model
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.rollout_length = rollout_length
        self.epochs_per_update = epochs_per_update
        self.minibatch_size = minibatch_size

        self.device = device
        self.expose_diagnostics = expose_diagnostics

        self.rng = random.Random(algorithm_seed)
        torch.manual_seed(algorithm_seed)

        if model == "mlp":
            make_backbone = lambda nf, out: Mlp(nf, out, list(hidden_sizes))
        elif model == "wesn":
            make_backbone = lambda nf, out: Wesn(nf, out, **(wesn or {}))
        else:
            raise ValueError(f"Unknown model: {model!r}")

        self.actors = [GaussianActor(make_backbone(num_features_per_sector[s], 1), init_log_std).to(device)
                       for s in range(num_sectors)]
        self.critics = [make_backbone(num_features_per_sector[s], 1).to(device)
                        for s in range(num_sectors)]
        self.optimizers = [
            torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=learning_rate)
            for actor, critic in zip(self.actors, self.critics)
        ]

        # Per-sector windowing history for _build_state -- ReplayBuffer
        # reused ONLY for its .recent() method (see module docstring);
        # small fixed capacity since .recent() never reads more than
        # sequence_length-1 entries back regardless of how large this is.
        buffer_sequence_length = sequence_length if model == "wesn" else None
        history_capacity = max(sequence_length * 4, 64)
        self.history_buffers = [ReplayBuffer(history_capacity, self.rng, sequence_length=buffer_sequence_length)
                                for _ in range(num_sectors)]

        # One open (act() -> observe()) transition per sector at a time,
        # same alternation invariant RLTiltController already relies on for
        # Dqn's own replay-buffer bookkeeping (decide() then learn() every
        # turn, never two opens before a close).
        self._pending = {}
        # Per-sector on-policy rollout: list of (state, raw_action,
        # log_prob, value, next_value, reward, done) tuples -- state is
        # [features] (mlp) or [sequence_length, features] (wesn), the exact
        # window act() used for that decision.
        self.rollouts = [[] for _ in range(num_sectors)]

        num_features_repr = (str(num_features_per_sector[0]) if len(set(num_features_per_sector)) == 1
                             else f"variable {min(num_features_per_sector)}-{max(num_features_per_sector)}")
        logger.info("Ppo constructed: model=%s num_sectors=%d num_features=%s num_actions=%d "
                   "action_grid=[%.2f, %.2f] step=%.3f rollout_length=%d %s device=%s",
                   model, num_sectors, num_features_repr, self.num_actions, self.theta_min, self.theta_max,
                   self.step, rollout_length,
                   f"sequence_length={sequence_length}" if model == "wesn" else "", device)

    def _quantize(self, raw_deg: float) -> int:
        """Continuous tilt [deg] -> floor-quantized action_grid_deg index."""
        clamped = min(max(raw_deg, self.theta_min), self.theta_max)
        idx = int(np.floor((clamped - self.theta_min) / self.step + 1e-9))
        return int(np.clip(idx, 0, self.num_actions - 1))

    def _build_state(self, sector, observations):
        """The [1, ...] network input for `sector`, or None if a real
        window can't be formed yet (wesn only -- fewer than
        sequence_length-1 real observations buffered) -- mirrors Dqn's own
        _build_state exactly (see drl/dqn.py), same warm-up-gap
        convention: too little history means this decision falls back to
        uniform random exploration (act()) and isn't learned from, same as
        Dqn's replay-buffer-backed warm-up."""
        if self.model == "wesn":
            buffer = self.history_buffers[sector]
            n_history = self.sequence_length - 1
            if len(buffer) < n_history:
                return None
            if n_history == 0:
                state_seq = observations[sector][None]
            else:
                history = buffer.recent(n_history)
                state_seq = np.concatenate([history, observations[sector][None]], axis=0)
            return torch.as_tensor(state_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        return torch.as_tensor(observations[sector], dtype=torch.float32, device=self.device).unsqueeze(0)

    def _shift_window(self, state, next_observation):
        """pending['state']'s window (numpy, no batch dim), advanced by one
        real observation -- builds the bootstrap value's own input
        directly from data already in hand, rather than depending on
        observe()'s own later history_buffers.add() call having already
        run this turn."""
        if self.model == "wesn":
            return np.concatenate([state[1:], np.asarray(next_observation, dtype=float)[None]], axis=0)
        return np.asarray(next_observation, dtype=float)

    def act(self, observations, training, mask=None, default_action=None):
        logger.function("Ppo.act start: training=%s", training)
        actions = (np.zeros(self.num_sectors, dtype=np.int64) if default_action is None
                  else np.array(default_action, dtype=np.int64).copy())
        self.last_decision_info = {}

        for sector in range(self.num_sectors):
            if mask is not None and not mask[sector]:
                logger.debug("Ppo.act: No meaningful observation, choosing default action")
                continue

            state_t = self._build_state(sector, observations)
            if state_t is None:
                # wesn only: not enough buffered history yet for a real
                # window -- uniform random exploration; no _pending entry,
                # since there's no real log_prob/value for an action that
                # didn't come from the current policy.
                raw_deg = self.rng.uniform(self.theta_min, self.theta_max)
                actions[sector] = self._quantize(raw_deg)
                logger.debug("Ppo.act: buffer < sequence length, choosing random action")
                continue

            with torch.no_grad():
                mean, std = self.actors[sector](state_t)
                mean, std = mean.squeeze(0), std.squeeze(0)
                if training:
                    dist = Normal(mean, std)
                    raw_action = dist.sample()
                    log_prob = float(dist.log_prob(raw_action).item())
                else:
                    # Deterministic eval: the actor's own mean, no
                    # sampling -- same convention as Dqn's epsilon=0 greedy.
                    raw_action = mean
                    log_prob = None
                raw_deg = float(raw_action.item())

                if training:
                    value = float(self.critics[sector](state_t).squeeze().item())
                    self._pending[sector] = dict(state=state_t.squeeze(0).cpu().numpy().copy(),
                                                 raw_action=raw_deg, log_prob=log_prob, value=value)

            actions[sector] = self._quantize(raw_deg)
            logger.debug("Ppo.act: sector=%d raw_deg=%.4f -> idx=%d", sector, raw_deg, actions[sector])

        logger.function("Ppo.act end")
        return actions

    def observe(self, observations, actions, rewards, next_observations, terminal, mask=None, training=True):
        logger.function("Ppo.observe start: terminal=%s training=%s", terminal, training)
        num_updated = 0

        for sector in range(self.num_sectors):
            if mask is not None and not mask[sector]:
                self._pending.pop(sector, None)
                continue

            # Keep the wesn windowing history current regardless of
            # training -- same reasoning as Dqn's own unconditional
            # buffer.add(): a stretch of eval-only calls shouldn't leave
            # _build_state's window stale once training resumes. Cheap
            # no-op consumer when model="mlp" (_build_state never reads
            # history_buffers then).
            self.history_buffers[sector].add(observations[sector], actions[sector], rewards[sector],
                                             next_observations[sector], float(terminal))

            if not training:
                self._pending.pop(sector, None)
                continue

            pending = self._pending.pop(sector, None)
            if pending is None:
                # act() wasn't called with training=True for this sector
                # this turn (e.g. it held its default action, or its
                # window wasn't warmed up yet) -- nothing real to credit.
                continue

            with torch.no_grad():
                next_state_np = self._shift_window(pending["state"], next_observations[sector])
                next_state_t = torch.as_tensor(next_state_np, dtype=torch.float32,
                                               device=self.device).unsqueeze(0)
                next_value = float(self.critics[sector](next_state_t).squeeze().item())

            self.rollouts[sector].append((
                pending["state"], pending["raw_action"], pending["log_prob"],
                pending["value"], next_value, float(rewards[sector]), bool(terminal),
            ))

            if len(self.rollouts[sector]) >= self.rollout_length:
                self._update(sector)
                self.rollouts[sector] = []
                num_updated += 1

        logger.debug("Ppo.observe: sectors_updated=%d", num_updated)
        logger.function("Ppo.observe end")

    def _update(self, sector):
        logger.function("Ppo._update start: sector=%d", sector)
        buf = self.rollouts[sector]
        states, raw_actions, old_log_probs, values, next_values, rewards, dones = zip(*buf)

        values = np.asarray(values, dtype=float)
        next_values = np.asarray(next_values, dtype=float)
        rewards = np.asarray(rewards, dtype=float)
        dones = np.asarray(dones, dtype=float)

        # GAE advantages + returns (critic's own regression target).
        advantages = np.zeros(len(buf), dtype=float)
        gae = 0.0
        for t in reversed(range(len(buf))):
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        raw_actions_t = torch.as_tensor(raw_actions, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        actor, critic, optimizer = self.actors[sector], self.critics[sector], self.optimizers[sector]
        indices = list(range(len(buf)))
        for _ in range(self.epochs_per_update):
            self.rng.shuffle(indices)
            for start in range(0, len(indices), self.minibatch_size):
                mb = indices[start:start + self.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)

                mean, std = actor(states_t[mb_t])
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(raw_actions_t[mb_t])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[mb_t])
                surr1 = ratio * advantages_t[mb_t]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t[mb_t]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = critic(states_t[mb_t]).squeeze(-1)
                value_loss = torch.nn.functional.mse_loss(value_pred, returns_t[mb_t])

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                if not torch.isfinite(loss):
                    logger.warning("Ppo._update: sector=%d non-finite loss=%s", sector, loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.step_losses.append(loss.item())
                self.step_loss_sectors.append(sector)

        logger.debug("Ppo._update: sector=%d rollout=%d mean_return=%.4f mean_advantage=%.4f",
                    sector, len(buf), float(returns.mean()), float(advantages.mean()))
        logger.function("Ppo._update end: sector=%d", sector)

    def save(self, path):
        logger.function("Ppo.save start: path=%s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actors": [actor.state_dict() for actor in self.actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "action_grid_deg": self.action_grid_deg,
        }, path)
        logger.info("Ppo: saved %d sector actor/critic pairs to %s", self.num_sectors, path)
        logger.function("Ppo.save end")
