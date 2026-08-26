"""PPO baseline config (runbook section 27).

Network sizes and PPO hyperparameters are mjlab defaults (128,128,128
hidden dims, standard PPO clip/GAE settings) — reasonable for a 21-dim
observation / 6-dim action task, unvalidated beyond "the smoke run doesn't
crash and the reward curve moves." Tune once real training data exists.

logger="tensorboard" instead of mjlab's own default ("wandb") because no
W&B account is configured for this project; avoids an external service
dependency for a first baseline.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def get_reaching_ppo_cfg(max_iterations: int = 1000) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="kinova_reach",
        logger="tensorboard",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=max_iterations,
    )
