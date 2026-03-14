# Dynamic Soaring RL

Train a reinforcement learning agent to perform **dynamic soaring** -- the technique used by albatrosses to fly thousands of kilometers without flapping their wings, by extracting energy from wind speed gradients near the ocean surface.

## How It Works

Dynamic soaring exploits the wind shear in the atmospheric boundary layer. Near the ocean surface, wind speed increases logarithmically with altitude. By repeatedly climbing into stronger wind (gaining airspeed) and descending into weaker wind (losing less airspeed), a bird can maintain or even gain total energy indefinitely.

This project uses **PPO (Proximal Policy Optimization)** via Stable-Baselines3 to learn this behavior from scratch.

## Architecture

```
src/dynamic_soaring/
├── config.py              # Centralized configuration (dataclasses + YAML)
├── physics/
│   ├── wind.py            # Logarithmic/power-law wind profiles
│   ├── aerodynamics.py    # Lift/drag with stall modeling
│   └── dynamics.py        # RK4 integrator, force summation
├── envs/
│   ├── soaring_env.py     # Gymnasium environment (13D obs, 2D action)
│   └── rewards.py         # Energy-based reward function
├── training/
│   ├── train.py           # SB3 PPO training pipeline
│   └── callbacks.py       # Custom TensorBoard metrics
├── evaluation/
│   ├── evaluate.py        # Policy evaluation & statistics
│   └── metrics.py         # Soaring cycle detection, energy analysis
└── visualization/
    ├── trajectory_3d.py   # PyVista interactive 3D visualization
    ├── training_curves.py # Training progress plots
    └── wind_field.py      # Wind profile visualization
```

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Train

```bash
python scripts/train.py --config configs/default.yaml
```

Override settings:
```bash
python scripts/train.py --config configs/default.yaml --timesteps 500000 --seed 123
```

### Monitor Training

```bash
tensorboard --logdir logs/
```

### Evaluate

```bash
python scripts/evaluate.py --model checkpoints/best_model.zip
```

### Visualize

```bash
# 3D trajectory (PyVista)
python scripts/visualize.py --model checkpoints/best_model.zip --mode trajectory

# Wind profile
python scripts/visualize.py --mode wind

# Training curves
python scripts/visualize.py --mode training

# Matplotlib fallback
python scripts/visualize.py --model checkpoints/best_model.zip --backend matplotlib
```

## Physics Model

- **Bird**: Wandering albatross (8.5 kg, 3.1m wingspan, AR=15)
- **Aerodynamics**: Lift with finite-wing correction and Viterna post-stall model, parabolic drag polar
- **Wind**: Logarithmic boundary layer profile `U(z) = U_ref * ln(z/z0) / ln(z_ref/z0)`
- **Integration**: 4th-order Runge-Kutta at dt=0.02s

## RL Setup

- **Algorithm**: PPO (Stable-Baselines3)
- **Observation** (13D): altitude, airspeed, ground speed, climb rate, flight path angle, heading, bank angle, AoA, wind speed, wind gradient, relative wind direction, specific energy, load factor
- **Action** (2D): angle of attack command, bank angle command (mapped from [-1,1] to physical ranges)
- **Reward**: energy gain + survival bonus - control smoothness penalty - crash/stall penalty

## Configuration

All parameters are in `configs/default.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wind.reference_speed` | 15.0 m/s | Wind at 10m height |
| `sim.dt` | 0.02 s | Physics timestep |
| `sim.max_episode_steps` | 3000 | Episode length (60s) |
| `training.total_timesteps` | 2M | Training budget |
| `training.n_envs` | 8 | Parallel environments |
| `training.net_arch` | [256, 256] | Policy network |

## Tests

```bash
pytest tests/ -v
```

## License

This project is open-source and available under the MIT License.
