# LunarLander-v3 — A2C Baseline

Minimal A2C agent using PyTorch, evaluated on a modified [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) (stronger wind, lower gravity).

## Run

Dependencies (Python 3.10, Torch 2.9.1, gymnasium[box2d]) are pinned in `pyproject.toml` and `.python-version`, so `uv` sets everything up on first run:

```bash
uv run python eval.py
```

## TODOs

1. `agent.py` — pick the action index from the policy logits at inference time.
2. `rl_model.py` (`collect_rollouts`) — sample an action from the policy during rollouts.
3. `rl_model.py` (`loss_a2c`) — compute discounted n-step returns (respecting `is_done`), the value and policy losses, and the entropy regularization term.

## Expected performance

After resolving the TODOs, training with

```bash
uv run python train.py --n_iterations=50000    # ~20 min
```

should reach **50–100** points on average.
