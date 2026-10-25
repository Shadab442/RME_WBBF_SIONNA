"""Lazy policy factory -- only imports torch when a torch-backed policy is
actually requested."""

from helpers.utils import get_logger

logger = get_logger(__name__)


def create_policy(name, num_sectors, num_features, num_actions, action_grid_deg=None,
                  dqn_kwargs=None, ppo_kwargs=None, algorithm_seed=0, device=None, expose_diagnostics=False):
    """Construct the selected algorithm.

    :param name: "random" | "dqn" | "ppo".
    :param action_grid_deg: [num_actions] candidate tilts [deg] -- required
        for "ppo" (needs the actual degree values to clamp/quantize its
        continuous action onto), ignored for "random"/"dqn" (they only need
        the count, via num_actions).
    :param dqn_kwargs: dict of Dqn hyperparameters (hidden_sizes,
        learning_rate, gamma, batch_size, replay_capacity,
        train_steps_per_interval, target_update_steps, epsilon_start,
        epsilon_end, epsilon_decay_rate); ignored for "random"/"ppo".
    :param ppo_kwargs: dict of Ppo hyperparameters (hidden_sizes,
        learning_rate, gamma, gae_lambda, clip_epsilon, entropy_coef,
        value_coef, rollout_length, epochs_per_update, minibatch_size,
        init_log_std); ignored for "random"/"dqn".
    :param device: "cuda:0" | "cpu" | ... -- defaults to CUDA if available,
        resolved lazily (only "dqn"/"ppo" need torch at all).
    :param expose_diagnostics: see Dqn's own constructor docstring;
        ignored for "random" (RandomPolicy has no Q-values to expose).
    """
    logger.function("create_policy start: name=%s num_sectors=%d num_features=%s num_actions=%d",
                    name, num_sectors, num_features, num_actions)

    if name == "random":
        from drl.random_policy import RandomPolicy
        logger.info("create_policy: selected policy=random")
        policy = RandomPolicy(num_sectors, num_actions, algorithm_seed=algorithm_seed)
        logger.function("create_policy end: name=random")
        return policy

    if name == "dqn":
        import torch
        from drl.dqn import Dqn
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("create_policy: selected policy=dqn model=%s device=%s",
                   (dqn_kwargs or {}).get("model", "wesn"), device)
        policy = Dqn(num_sectors, num_features, num_actions,
                    algorithm_seed=algorithm_seed, device=device,
                    expose_diagnostics=expose_diagnostics,
                    **(dqn_kwargs or {}))
        logger.function("create_policy end: name=dqn")
        return policy

    if name == "ppo":
        import torch
        from drl.ppo import Ppo
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if action_grid_deg is None:
            raise ValueError("create_policy: action_grid_deg is required for policy=ppo")
        logger.info("create_policy: selected policy=ppo device=%s", device)
        policy = Ppo(num_sectors, num_features, action_grid_deg,
                    algorithm_seed=algorithm_seed, device=device,
                    expose_diagnostics=expose_diagnostics,
                    **(ppo_kwargs or {}))
        logger.function("create_policy end: name=ppo")
        return policy

    logger.warning("create_policy: unknown policy name=%s", name)
    raise ValueError(f"Unknown policy: {name!r}")
