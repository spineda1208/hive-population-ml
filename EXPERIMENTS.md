# Experiment Results

## Overview

This document summarizes the results of comparing three model architectures on the **population classification task** (predicting whether a hive has >= 20 frames of bees).

## Test Set Results

| Model | Accuracy | Balanced Accuracy | F1 Score | AUC-ROC |
|-------|----------|-------------------|----------|---------|
| **MLP** | **83.7%** | **80.5%** | **0.881** | **0.926** |
| Transformer | 67.4% | 72.2% | 0.708 | 0.637 |
| LSTM | 67.4% | 50.0% | 0.806 | 0.539 |

## Experimental Setup

### Data
- Dataset: MSPB (Multi-modal Sensor dataset with Phenotypic trait measurements from honey Bees)
- Task: Binary classification (population >= 20 frames of bees)
- Split: By hive ID (70/15/15 train/val/test) to prevent data leakage
- Window: 7-day temporal windows centered on annotation dates

### Model Configurations

**MLP (Aggregated Features)**
- Input: 243 statistical features (mean, std, min, max, delta for each sensor)
- Hidden dim: 128
- Layers: 3
- Parameters: 82,050

**LSTM (Sequence Model)**
- Input: Raw sensor sequences (22 features, hourly sampling)
- Hidden dim: 64
- Layers: 2 (bidirectional)
- Attention pooling
- Parameters: 184,195

**Transformer (Sequence Model)**
- Input: Raw sensor sequences (22 features, hourly sampling)
- d_model: 64
- Layers: 2
- Heads: 4
- Parameters: 103,778

### Training
- Optimizer: AdamW
- Learning rate: 1e-4 (MLP), 1e-3 (LSTM/Transformer)
- Weight decay: 0.01
- Early stopping patience: 15-20 epochs
- LR scheduler: ReduceLROnPlateau

## Key Findings

### 1. MLP significantly outperforms sequence models

The MLP using aggregated statistical features achieves much better performance than both LSTM and Transformer models working on raw time-series data. This suggests that hand-crafted temporal statistics effectively capture the discriminative patterns in the sensor data.

### 2. LSTM failed to learn

The LSTM completely collapsed to predicting the majority class (balanced accuracy = 50%), despite trying multiple hyperparameter configurations including:
- Different learning rates (1e-4, 1e-3)
- Different hidden dimensions (64, 128)
- Sequence subsampling to reduce length

This suggests the raw sensor sequences don't have easily learnable temporal patterns for this task, or require more sophisticated preprocessing.

### 3. Transformer shows partial learning

The Transformer performed better than LSTM (72% balanced accuracy) but still falls short of MLP. It demonstrates the model is learning some patterns but not as effectively as hand-crafted statistical features.

### 4. Feature engineering matters

The aggregated features (243 features including temporal statistics) capture the discriminative patterns much better than letting neural networks learn directly from sequences. This is consistent with findings in the MSPB paper.

## Comparison to MSPB Paper Baseline

- Paper reports ~65.8% accuracy baseline for population classification
- Our MLP achieves **83.7%** accuracy - a substantial improvement of ~18 percentage points

## Saved Results

All experiment results are saved in `results/`:
- `mlp_population_exp1/` - Best model checkpoint and training history
- `lstm_population_exp2/` - LSTM experiment  
- `transformer_population_exp1/` - Transformer experiment

Each directory contains:
- `config.json` - Hyperparameters used
- `results.json` - Training history and test metrics
- `checkpoints/best_model.pt` - Best model weights

## Recommendations for Future Work

1. **Stick with MLP for production** - The aggregated features approach works well and is computationally efficient

2. **Try hybrid approaches** - Use statistical features as additional input to sequence models

3. **Better sequence preprocessing** - Consider:
   - Learned temporal embeddings
   - 1D convolutional preprocessing before LSTM/Transformer
   - Multi-scale temporal aggregation

4. **Expand to other tasks** - Test on:
   - Honey yield prediction (regression)
   - Varroa infestation (regression)
   - Winter mortality prediction (classification)

5. **Ensemble methods** - Combine MLP with sequence models for potentially better generalization

---

## Phenotypic Task Experiments

Following the recommendations from the population classification experiments, we extended the framework to support additional phenotypic traits: **Varroa infestation** and **honey yield prediction**.

### Multi-Scale Temporal Features

We implemented multi-scale temporal aggregation to better capture trends:
- **Weekly statistics**: Trend across 7-day windows (std, range of weekly means)
- **Monthly statistics**: Trend across 30-day windows
- **Early vs late comparison**: Detects trajectory changes within the observation period

This increased feature dimensionality from 110 to **308 features**.

### Varroa Infestation Prediction

**Dataset**: 53 samples (extremely limited)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Lasso** | **0.410** | **0.519** | **-0.047** |
| Ridge | 0.418 | 0.531 | -0.091 |
| RandomForest | 0.424 | 0.553 | -0.185 |
| GradientBoosting | 0.436 | 0.557 | -0.200 |
| Ensemble_Weighted | 0.417 | 0.535 | -0.109 |
| MLP (baseline) | 0.475 | 0.548 | -0.226 |

**Finding**: All models fail to predict Varroa infestation (negative R² indicates worse than mean prediction). The dataset is too small (53 samples, 308 features) and the target variable has high variance that cannot be explained by sensor data alone.

### Honey Yield Prediction

**Dataset**: 46 samples

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **RandomForest** | **10.26** | **13.12** | **0.372** |
| GradientBoosting | 10.27 | 14.10 | 0.274 |
| Ensemble_Weighted | 10.08 | 13.40 | 0.345 |
| Ridge | 11.52 | 14.98 | 0.181 |
| Lasso | 12.71 | 15.83 | 0.086 |

**Finding**: RandomForest achieves R² = 0.37, explaining ~37% of honey yield variance from sensor data. This is a meaningful signal despite the small dataset.

### Key Insights from Phenotypic Experiments

1. **Small datasets favor traditional ML**: With only 46-53 samples, deep learning (MLP) underperforms compared to RandomForest and regularized linear models.

2. **Varroa is unpredictable from sensors alone**: Varroa infestation levels likely depend on factors not captured by hive sensors (e.g., treatment history, nearby apiaries, seasonal mite reproduction cycles).

3. **Honey yield shows promise**: The positive R² suggests sensor data does contain information about honey production potential, likely through activity patterns and environmental conditions.

4. **Multi-scale features help**: The temporal trend features provide richer representations than simple statistics, though more data is needed to fully leverage them.

### Saved Results

Phenotypic experiment results in `results/`:
- `mlp_varroa_baseline/` - MLP baseline on Varroa
- `mlp_varroa_regularized/` - MLP with stronger regularization
- `ensemble_varroa_*/` - Ensemble methods on Varroa
- `ensemble_honey_yield_*/` - Ensemble methods on honey yield

---

## Reproducing Results

### Population Classification (Best: MLP)

```bash
# MLP (best performing)
uv run python scripts/train.py --model mlp --task population --task-type classification \
    --epochs 100 --batch-size 32 --learning-rate 1e-4 --hidden-dim 128 --num-layers 3

# LSTM
uv run python scripts/train.py --model lstm --task population --task-type classification \
    --epochs 100 --batch-size 32 --learning-rate 1e-3 --hidden-dim 64 --num-layers 2

# Transformer
uv run python scripts/train.py --model transformer --task population --task-type classification \
    --epochs 100 --batch-size 32 --learning-rate 1e-3 --hidden-dim 64 --num-layers 2
```

### Phenotypic Tasks (Ensemble)

```bash
# Varroa prediction (ensemble)
uv run python scripts/train_ensemble.py --task varroa

# Honey yield prediction (ensemble)
uv run python scripts/train_ensemble.py --task honey_yield
```
