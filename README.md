# Policy-Based Reinforcement Learning with Vision Transformer for Multi-Modal Brain Lesion Localization

This repository contains the project documentation and baseline Reinforcement Learning (RL) implementations for multi-modal MRI brain tumor localization on the BraTS dataset benchmark.

## Repository Structure

- **`main` Branch**: Contains project design documentation, technical walkthroughs, implementation plans, and benchmark visual results:
  - [`walkthrough.md`](walkthrough.md): Complete technical assessment, empirical results briefing, and bounding box visualizations.
  - [`implementation_plan.md`](implementation_plan.md): Architectural design document.
  - [`docs/`](docs/): Visual benchmark figures and table infographics.

- **`nirranjan121` Branch**: Contains the complete runnable baseline RL code:
  - `dataset.py`: BraTS 2D slice HDF5 data loader with multi-modal 4-channel stacking (T1, T1ce, T2, FLAIR).
  - `environment.py`: Gym-like `BrainLesionEnv` environment for bounding box ROI transformations.
  - `model.py`: 3-layer CNN feature encoder fused with spatial box coordinate Q-network.
  - `agent_dqn.py`: Standard DQN and Double-DQN (DDQN) agents with experience replay buffer.
  - `train.py`: Baseline training pipeline for DQN and Double-DQN over episodes.
  - `evaluate.py`: Benchmark evaluation script computing Dice, IoU, Sensitivity, Precision, and Convergence speed.
  - `test_visualize.py`: Standalone testing and visual bounding box renderer overlaying GT (Green), DDQN (Cyan), and DQN (Red) boxes.
  - `app.py` & `templates/index.html`: Interactive Web Visual Dashboard.

## Quick Start (on `nirranjan121` branch)

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python train.py --episodes 40 --samples 50

# Run standalone testing & generate bounding box result images
python test_visualize.py

# Launch interactive web visual dashboard
python app.py
```
