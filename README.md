# Brain Lesion Localization via Reinforcement Learning — CNN vs. ViT Encoder Study

> **Academic Project**: Explainable Policy-Based Reinforcement Learning with Vision Transformer for Multi-Modal Brain Lesion Localization

An RL agent learns to localize brain tumors by sequentially adjusting a 2D spatial view window (move, zoom, terminate) over multi-modal MRI scans, rather than performing single-pass pixel segmentation. This repository currently compares two visual feature encoders — a ResNet-style **CNN** and a **Vision Transformer (ViT)** — plugged into an otherwise identical DQN/Double-DQN agent, environment, and evaluation pipeline.

---

## 1. Current Status at a Glance

| Component | Status |
|---|---|
| Data pipeline, RL environment, reward shaping | **Built and working** |
| Unified DQN agent (CNN or ViT encoder, optional Double-DQN) | **Built and working** |
| Evaluation metrics (Dice/IoU/Sensitivity/Precision, per ET/TC/WT) | **Built and working** |
| Qualitative trajectory visualization | **Built and working** |
| PowerPoint report generator | **Built** (see caveat in §6 — current deck uses placeholder numbers) |
| CNN-encoder model training | **Partially done** — only 50 of the configured 1000 episodes have been run |
| CNN-encoder model evaluation | **Done** — full 125-case validation split evaluated |
| ViT-encoder model training | **Not started** — `outputs/vit_dqn/` is empty |
| CNN vs. ViT comparison report/plot | **Not generated yet** — requires the ViT model to be trained first |
| `requirements.txt` | **Missing** — not yet committed to the repo |

In short: the full pipeline (data → environment → agent → training → evaluation → visualization → presentation) is implemented end-to-end and has been proven out on the CNN encoder, but only a short, non-converged training run has actually been executed, and the ViT side of the comparison has not been trained or evaluated at all yet.

---

## 2. Dataset

- **Dataset**: ASNR-MICCAI BraTS2023 Multi-Modal Glioma Challenge (BraTS2023)
- **Modalities**: `t1n` (native T1), `t1c` (post-contrast T1/T1ce), `t2w` (T2-weighted), `t2f` (FLAIR)
- **Ground-truth subregions**: `WT` (Whole Tumour, labels 1/2/3), `TC` (Tumour Core, labels 1/3), `ET` (Enhancing Tumour, label 3)
- **Split** (`data/splits/train_val_split.json`, reproducible via seed 42): 50% subsample of the 1,251 labeled training cases (625 cases) → **500 train / 125 validation**.
- The official unlabeled `ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData` (219 cases) is present on disk but unused, since Dice/IoU require ground-truth masks.

---

## 3. Repository Structure

```
final_year_project/
├── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/     # raw BraTS2023 NIfTI volumes (1,251 cases)
├── ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/   # official unlabeled val set (unused for eval)
├── configs/
│   ├── cnn_dqn.yaml          # DQN + CNN encoder
│   └── vit_dqn.yaml          # DQN + ViT encoder
├── data/splits/train_val_split.json
├── preprocessing/
│   ├── nifti_loader.py          # BraTS case discovery + NIfTI loading
│   ├── normalization.py         # z-score intensity normalization
│   ├── resampling.py            # 1mm isotropic spacing check/resample
│   └── preprocessing_pipeline.py
├── environment/
│   ├── brain_lesion_env.py      # BrainLesionEnv (Gymnasium), agent-agnostic
│   ├── actions.py                # 7 discrete actions
│   ├── rewards.py                # Dice-based differential reward
│   └── state.py                  # bbox update logic
├── models/
│   ├── cnn_encoder.py           # ResNet-style CNN encoder
│   ├── vit_encoder.py           # Vision Transformer encoder (patch embed + MHSA)
│   └── q_network.py             # unified Q-network, selects encoder via config
├── agents/
│   ├── replay_buffer.py
│   └── dqn_agent.py              # DQN, with Double-DQN as a config toggle
├── training/
│   └── train.py                  # single training loop for either encoder
├── evaluation/
│   ├── metrics.py                # Dice/IoU/Sensitivity/Precision, per-subregion
│   ├── evaluate.py               # evaluation runner (best-Dice-window scoring)
│   ├── visualize.py              # per-episode trajectory overlay plots
│   └── compare_models.py         # CNN vs. ViT comparison report + plot generator
├── utils/split_dataset.py        # reproducible train/val split generator
├── outputs/
│   ├── cnn_dqn/{checkpoints,logs,plots}/   # CNN run artifacts (partial)
│   ├── vit_dqn/{checkpoints,logs,plots}/   # ViT run artifacts (empty — not yet run)
│   ├── comparison/                          # cnn_eval_summary.json only so far
│   └── presentation/                        # generated .pptx (placeholder numbers, see §6)
├── generate_presentation.py      # builds a PowerPoint report from comparison_summary.json
└── train.py                      # root CLI entrypoint
```

---

## 4. RL Environment, Action Space, and Reward

`environment/brain_lesion_env.py::BrainLesionEnv` is a Gymnasium-style environment, independent of any specific agent implementation.

- **State**: a `[4, 64, 64]` multi-modal image crop plus the normalized bounding box `[xmin, ymin, xmax, ymax]`.
- **Actions** (7, discrete): `MOVE_LEFT`, `MOVE_RIGHT`, `MOVE_UP`, `MOVE_DOWN`, `ZOOM_IN`, `ZOOM_OUT`, `TERMINATE`.
- **Episode**: starting window covers 50% of the slice (`initial_scale=0.5`), placed at a randomized position during training and centered during validation; `max_steps=20`.
- **Reward** (`environment/rewards.py`): `10 × (Dice_t − Dice_{t-1}) − 0.01` per step; on `TERMINATE`, `+2.0` if `Dice ≥ 0.3` else `−0.1`.

---

## 5. Model Architecture

Both encoders share the same downstream head and are selected purely via the `encoder_type` config field, so training/agent code is identical between runs:

- **CNN encoder** (`models/cnn_encoder.py`): a small ResNet-style stack (strided conv stem → 2 residual blocks → global average pool) mapping `[4, 64, 64]` → a `feature_dim`-length vector.
- **ViT encoder** (`models/vit_encoder.py`): 8×8 patch embedding (64 patches + CLS token) → 4-layer Transformer encoder (4 heads, GELU, dropout 0.1) → CLS token projected to `feature_dim`.
- **Fusion** (`models/q_network.py::QNetwork`): the encoder's feature vector is concatenated with a small MLP embedding of the 4 normalized bbox coordinates, then passed through a 2-layer MLP head to produce 7 Q-values.
- **Agent** (`agents/dqn_agent.py::DQNAgent`): standard replay-buffer DQN; `use_double_dqn: true` in both configs enables the decoupled Double-DQN target (`y = r + γ·Q_target(s', argmax_a' Q_online(s', a'))`).

---

## 6. Results So Far

### CNN-encoder run (`outputs/cnn_dqn/`)

Only **50 of the configured 1000 episodes** were actually run (`outputs/cnn_dqn/logs/training_history.json` contains 50 entries) — this is an early, non-converged checkpoint, not a finished training result. The resulting `model_best.pt` was evaluated on the **full 125-case validation split** with best-Dice-window scoring (`outputs/comparison/cnn_eval_summary.json`):

| Subregion | Dice | IoU | Sensitivity | Precision |
|---|---|---|---|---|
| WT (Whole Tumour) | 0.2655 ± 0.0934 | 0.1564 ± 0.0624 | 0.8507 | 0.1613 |
| TC (Tumour Core) | 0.1137 ± 0.0826 | 0.0624 ± 0.0487 | 0.8694 | 0.0631 |
| ET (Enhancing Tumour) | 0.0768 ± 0.0543 | 0.0408 ± 0.0303 | 0.8612 | 0.0411 |

Mean best-step fraction: 75.4% of the episode — i.e. the best window is typically found late, close to the step budget running out.

These numbers should be read as a **sanity-check baseline from a short run**, not a converged result — full-length training (1000 episodes, matching the config) has not yet been executed.

### ViT-encoder run

**Not started.** `outputs/vit_dqn/checkpoints/`, `outputs/vit_dqn/logs/`, and `outputs/vit_dqn/plots/` are all empty. No `model_best.pt` exists to evaluate.

### Comparison report and presentation — important caveat

`evaluation/compare_models.py` (CNN vs. ViT report + bar-chart generator) and `generate_presentation.py` (PowerPoint builder) are both fully implemented, and a deck already exists at `outputs/presentation/Lesion_Localization_ViT_vs_CNN.pptx`. **However**, `outputs/comparison/comparison_summary.json` does not exist yet (only the standalone `cnn_eval_summary.json` does), so `generate_presentation.py` fell back to its hardcoded placeholder numbers (e.g. `vit_wt_dice = 0.4420`) rather than real evaluation output. **The existing .pptx does not reflect actual ViT results and must be regenerated** once `evaluation.compare_models.compare_cnn_vs_vit()` has been run against a trained ViT checkpoint.

---

## 7. How to Run

```bash
# Generate the reproducible train/val split (50% subsample) — already present, regenerate if needed
python -m utils.split_dataset

# Train either encoder (episodes come from configs/{cnn,vit}_dqn.yaml; override with --episodes)
python train.py --model-type cnn
python train.py --model-type vit

# Evaluate a trained checkpoint on the full validation split
python -m evaluation.evaluate --model-type cnn --output outputs/comparison/cnn_eval_summary.json
python -m evaluation.evaluate --model-type vit --output outputs/comparison/vit_eval_summary.json

# Generate the CNN vs. ViT comparison report + bar chart (runs both evaluations itself)
python -m evaluation.compare_models

# Rebuild the presentation deck from real comparison results
python generate_presentation.py
```

---

## 8. What's Left To Do

1. **Finish CNN training** — run the full 1000 episodes per `configs/cnn_dqn.yaml` (currently stopped at 50) and re-evaluate.
2. **Train the ViT encoder** — `configs/vit_dqn.yaml` is ready; no run has been started.
3. **Generate a real comparison report** — run `evaluation.compare_models.compare_cnn_vs_vit()` once both checkpoints exist, producing `comparison_summary.json`, `model_comparison.md`, and `metrics_comparison.png`.
4. **Regenerate the presentation** — `generate_presentation.py` currently ships a deck built on placeholder numbers; rerun it once step 3 is done.
5. **Add `requirements.txt`** — the repo currently has no pinned dependency list (known dependencies from imports: `torch`, `gymnasium`, `nibabel`, `numpy`, `matplotlib`, `pyyaml`, `python-pptx`).
6. **Investigate TERMINATE usage and reward calibration** for the current run once a longer training run is available — the 50-episode run is too short to diagnose behavioral issues like premature/never-terminating episodes.
