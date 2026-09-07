"""Lazy policy factory -- only imports torch when a torch-backed policy is
actually requested."""

from helpers.utils import get_logger

logger = get_logger(__name__)


def create_policy(name, num_sectors, num_features, num_actions,
                  dqn_kwargs=None, algorithm_seed=0, device=None):
    """Construct the selected algorithm.

    :param name: "random" | "dqn".
    :param dqn_kwargs: dict of Dqn hyperparameters (hidden_sizes,
        learning_rate, gamma, batch_size, replay_capacity,
        train_steps_per_interval, target_update_steps, epsilon_start,
        epsilon_end, epsilon_decay_rate); ignored for "random".
    :param device: "cuda:0" | "cpu" | ... -- defaults to CUDA if available,
        resolved lazily (only "dqn" needs torch at all).
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
                    **(dqn_kwargs or {}))
        logger.function("create_policy end: name=dqn")
        return policy

    logger.warning("create_policy: unknown policy name=%s", name)
    raise ValueError(f"Unknown policy: {name!r}")
