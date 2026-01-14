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

## Reproducing Results

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
