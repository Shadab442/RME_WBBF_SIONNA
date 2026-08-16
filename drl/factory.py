"""Lazy policy factory -- only imports torch when a torch-backed policy is
actually requested."""


def create_policy(name, num_sectors, num_features, num_actions,
                  dqn_kwargs=None, algorithm_seed=0, device="cpu"):
    """Construct the selected algorithm.

    :param name: "random" | "independent-dqn".
    :param dqn_kwargs: dict of IndependentDqn hyperparameters (hidden_sizes,
        learning_rate, gamma, batch_size, replay_capacity, warmup_steps,
        train_steps_per_interval, target_update_steps, epsilon_start,
        epsilon_end, epsilon_decay_rate); ignored for "random".
    """
    if name == "random":
        from drl.random_policy import RandomPolicy
        return RandomPolicy(num_sectors, num_actions, algorithm_seed=algorithm_seed)
    if name == "independent-dqn":
        from drl.independent_dqn import IndependentDqn
        return IndependentDqn(num_sectors, num_features, num_actions,
                              algorithm_seed=algorithm_seed, device=device,
                              **(dqn_kwargs or {}))
    raise ValueError(f"Unknown policy: {name!r}")
