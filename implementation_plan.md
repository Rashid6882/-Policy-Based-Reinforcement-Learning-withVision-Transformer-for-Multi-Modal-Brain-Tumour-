# Implementation Plan - DQN & Double-DQN Baseline for Multi-Modal Brain Lesion Localization

This project builds a complete baseline reinforcement learning framework implementing **DQN (Deep Q-Network)** and **Double-DQN (DDQN)** agents for multi-modal MRI brain lesion localization using the **BraTS** benchmark. It provides a benchmark to compare against the main project architecture (**SAC + Vision Transformer**).

## User Review Required

> [!IMPORTANT]
> - **Dataset Format**: The system utilizes the preprocessed BraTS 2D HDF5 slices (`(240, 240, 4)` images with T1, T1ce, T2, FLAIR modalities and `(240, 240, 3)` sub-region masks) located in `BraTS2020_training_data/content/data/`.
> - **Action Space & Bounding Box RL**: The agent operates as a sequential decision-making system. Given a 4-channel ROI patch and bounding box state, the discrete actions allow moving (Left, Right, Up, Down), scaling (Zoom In, Zoom Out), and triggering a terminal prediction (`Trigger/Stop`).
> - **Baselines**: Both standard **DQN** and **Double-DQN** are implemented with identical environment dynamics, reward functions, and feature encoders to isolate the effect of overestimation bias reduction.

## Open Questions

> [!NOTE]
> None at present. Default parameters (64x64 cropped ROI, 7 discrete actions, IoU step reward + Dice terminal reward) are configured for fast CPU/GPU training and evaluation.

## Proposed Changes

Grouping all baseline modules logically:

---

### 1. Dataset & Data Processing Module

#### [NEW] [dataset.py](file:///c:/Users/Win%2010/Documents/finalyearproject/dataset.py)
- Implements `BraTSDataset` to load `.h5` files containing 4-channel MRI modalities (T1, T1ce, T2, FLAIR) and ground-truth tumor masks.
- Extracts ground-truth bounding box coordinates $[x_{min}, y_{min}, x_{max}, y_{max}]$ from tumor sub-region masks (Whole Tumor / Tumor Core / Enhancing Tumor).
- Filters non-empty lesion slices for training and evaluation.

---

### 2. Reinforcement Learning Environment

#### [NEW] [environment.py](file:///c:/Users/Win%2010/Documents/finalyearproject/environment.py)
- Implements `BrainLesionEnv` (Gym interface):
  - **State**: Cropped multi-modal ROI patch $(4, 64, 64)$ + Normalized relative bounding box $[x_1/W, y_1/H, x_2/W, y_2/H, t/T_{max}]$.
  - **Actions**: 7 Discrete actions: `0: Move Left`, `1: Move Right`, `2: Move Up`, `3: Move Down`, `4: Zoom In`, `5: Zoom Out`, `6: Stop/Trigger`.
  - **Reward Function**: Differential IoU reward $r_t = (IoU_t - IoU_{t-1}) - \eta$ plus terminal Dice bonus upon `Stop` or max steps.
  - **Metrics Computation**: Calculates Dice coefficient, IoU (Jaccard index), Sensitivity (Recall), and Precision.

---

### 3. Model Architecture & Agents

#### [NEW] [model.py](file:///c:/Users/Win%2010/Documents/finalyearproject/model.py)
- Implements `CNNEncoder`: Deep Convolutional Feature Extractor with 3 Conv2D layers, BatchNorm, and ReLU activations for 4-channel ROI patches.
- Implements `DQNNetwork`: Combines CNN ROI features with spatial box coordinate features to output action Q-values $Q(s, a)$.

#### [NEW] [agent_dqn.py](file:///c:/Users/Win%2010/Documents/finalyearproject/agent_dqn.py)
- Implements `ReplayBuffer` for experience replay.
- Implements `DQNAgent`: Standard Deep Q-Network with $\epsilon$-greedy exploration decay and target network updating.
- Implements `DoubleDQNAgent`: Double-DQN algorithm decoupling action selection from target Q-value evaluation.

---

### 4. Training, Evaluation & Benchmarking Pipelines

#### [NEW] [train.py](file:///c:/Users/Win%2010/Documents/finalyearproject/train.py)
- Main training loop for DQN and Double-DQN agents over episodes.
- Saves model checkpoints, loss curves, episode rewards, and evaluation metrics.

#### [NEW] [evaluate.py](file:///c:/Users/Win%2010/Documents/finalyearproject/evaluate.py)
- Benchmark script to evaluate trained DQN & Double-DQN agents on validation slices.
- Generates metrics summary table (Dice, IoU, Sensitivity, Precision, Convergence Speed) and stores trajectory data for visualization.

---

### 5. Verification & Visual Dashboard

#### [NEW] [tests/test_pipeline.py](file:///c:/Users/Win%2010/Documents/finalyearproject/tests/test_pipeline.py)
- Unit tests verifying dataset loading, environment dynamics, Q-network tensor shapes, replay buffer sampling, and metric computations.

#### [NEW] [app.py](file:///c:/Users/Win%2010/Documents/finalyearproject/app.py)
- Flask/Web visual dashboard server providing interactive side-by-side visualization of:
  - Multi-modal MRI scan slices (T1, T1ce, T2, FLAIR).
  - Ground truth lesion bounding boxes vs DQN and Double-DQN predicted bounding box trajectories.
  - Quantitative comparative metric graphs and training curves.

#### [NEW] [templates/index.html](file:///c:/Users/Win%2010/Documents/finalyearproject/templates/index.html)
- Modern visual dashboard UI.

---

## Verification Plan

### Automated Tests
- Run `pytest` or `python -m unittest tests/test_pipeline.py` to verify environment step mechanics, reward calculations, model forward passes, and training step execution.
- Run `python train.py --episodes 10` to verify full training workflow without errors.
- Run `python evaluate.py` to verify metric computation and trajectory generation.

### Manual Verification & Visual Dashboard
- Launch `python app.py` and view the visual dashboard displaying bounding box step trajectories, ground truth masks, and comparative metrics between DQN and Double-DQN.
