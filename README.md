# ObjectRL

[![docs](https://readthedocs.org/projects/objectrl/badge/?version=latest)](https://objectrl.readthedocs.io/en/latest/)
[![license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/adinlab/objectrl/blob/master/LICENSE)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

<p align="center">
  <img src="docs/_static/imgs/logo.svg" alt="ObjectRL Logo" height="150">
</p>

**ObjectRL** is a deep reinforcement learning library designed for research and rapid prototyping. It focuses on deep actor-critic algorithms for continuous control tasks such as those in the MuJoCo environment suite, while providing a flexible object-oriented architecture that supports future extensions to value-based and discrete-action methods.

---

## Features

- Object-oriented design for easy experimentation  
- Implements popular deep RL algorithms for continuous control  
- Includes experimental implementations of Bayesian and value-based methods  
- Supports easy configuration via CLI and YAML files  
- Rich examples and tutorials for customization and advanced use cases  

---

## Supported Algorithms

- **DDPG** (Deep Deterministic Policy Gradient)  
- **TD3** (Twin Delayed DDPG)  
- **SAC** (Soft Actor-Critic)  
- **PPO** (Proximal Policy Optimization)  
- **REDQ** (Randomized Ensemble Double Q-Learning)  
- **DRND** (Distributional Random Network Distillation)  
- **OAC** (Optimistic Actor-Critic)  
- **PBAC** (PAC-Bayesian Actor-Critic)  
- **BNN-SAC** (Bayesian Neural Network SAC) — experimental, in examples  
- **DQN** (Deep Q-Network) — experimental, in examples  

---

## Installation

### Create Environment

```bash
conda create -n objectrl python=3.12 -y
conda activate objectrl
```

### Using PyPI

```bash
pip install objectrl
```

### From Source

```bash
git clone https://github.com/adinlab/objectrl.git
cd objectrl
pip install -e .
```

### Optional Dependencies

To enable additional features such as documentation generation:

```bash
pip install objectrl[docs]
```

---

## Quick Start Guide

Run your first experiment using Soft Actor-Critic (SAC) on the default `cheetah` environment:

If installed from PyPI:

```bash
python -m objectrl.main --model.name sac
```

If running from a cloned repo:

```bash
python objectrl/main.py --model.name sac
```

Other examples will assume running from a cloned repo.

### Change Algorithm and Environment

Run DDPG on the `hopper` environment:

```bash
python objectrl/main.py --model.name ddpg --env.name hopper
```

### Customize Training Parameters

Train SAC for 100,000 steps and evaluate every 5 episodes:

```bash
python objectrl/main.py --model.name sac --env.name hopper --training.max_steps 100000 --training.eval_episodes 5
```

### Use YAML Configuration Files

For more complex or reproducible setups, create YAML config files in `objectrl/config/model_yamls/` and specify them at runtime:

```bash
python objectrl/main.py --config objectrl/config/model_yamls/ppo.yaml
```

Example `ppo.yaml`:

```yaml
model:
  name: ppo
training:
  warmup_steps: 0
  learn_frequency: 2048
  batch_size: 64
  n_epochs: 10
```

---

## Need Help?

If you encounter common issues or errors during installation or usage, please see the [Issues](ISSUES.md) guide for solutions and tips.

For other questions or to report bugs, visit our [GitHub Issues page](https://github.com/adinlab/objectrl/issues).

---

## Documentation

Explore detailed documentation, tutorials, and API references at: [https://objectrl.readthedocs.io](https://objectrl.readthedocs.io)

---

## Citation

If you use ObjectRL in your research, please cite:

```bibtex
@article{baykal2025objectrl,
  title={ObjectRL: An Object-Oriented Reinforcement Learning Codebase}, 
  author={Baykal, Gulcin and  Akg{\"u}l, Abdullah and Haussmann, Manuel and Tasdighi, Bahareh and Werge, Nicklas and Wu Yi-Shan and Kandemir, Melih},
  year={2025},
  journal={arXiv preprint arXiv:2507.03487}
}
```




