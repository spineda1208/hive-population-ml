# Hive Population ML

Multi-modal machine learning models for honey bee hive phenotypic trait prediction using the MSPB (Multi-modal Sensor dataset with Phenotypic trait measurements from honey Bees) dataset.

## Overview

This project implements deep learning models to predict various honey bee colony traits from multi-modal sensor data including:
- Audio features (hive power, frequency band coefficients)
- Temperature
- Relative humidity

### Prediction Targets

| Task | Type | Description |
|------|------|-------------|
| Population (FoB) | Regression + Classification | Frames of bees count |
| Honey yield | Regression | Honey production in kg |
| Varroa infestation | Regression | Mites per 100 bees |
| Defensive behavior | Regression | Number of stings |
| Hygienic behavior | Regression | Cleaning capacity % |
| Winter mortality | Classification | Binary survived/failed |

## Design Decisions

### 1. Temporal Aggregation Strategy

**Decision**: Aggregate sensor data over 7-day windows centered on annotation dates.

**Rationale**: 
- Annotations are collected bi-weekly by experts
- Using a 7-day window (3 days before + annotation day + 3 days after) captures relevant temporal patterns
- This matches the approach used in the original MSPB paper

### 2. Feature Engineering

**Decision**: Extract statistical features at multiple temporal levels.

**Features computed**:
- Mean, standard deviation
- Min, max, range
- 1st and 2nd order deltas (temporal derivatives)
- Skewness, kurtosis (for aggregated windows)

### 3. Model Architectures

Three architectures are implemented for comparison:

#### A. MLP Baseline
- Simple feedforward network on aggregated features
- Fast training, good baseline for comparison
- Input: Flattened aggregated features

#### B. LSTM/GRU
- Bidirectional recurrent network for temporal sequences
- Handles variable-length sequences with attention pooling
- Captures temporal dependencies within observation windows

#### C. Transformer
- Multi-head self-attention over temporal sequences
- Separate embeddings for audio, temperature, humidity modalities
- Cross-modal fusion through attention layers
- Most expressive but requires more data

### 4. Multi-Task Learning

**Decision**: Shared backbone with task-specific heads.

**Rationale**:
- Phenotypic traits are correlated (e.g., honey yield correlates with population)
- Shared representations can improve generalization
- Auxiliary tasks provide additional supervision signal

### 5. Train/Val/Test Split Strategy

**Decision**: Split by hive ID, not by time.

**Rationale**:
- Prevents data leakage from temporal autocorrelation
- Models must generalize to unseen hives
- 70/15/15 split ratio

### 6. Handling Class Imbalance

**Decision**: Use weighted loss functions and stratified sampling.

**Methods**:
- Inverse frequency weighting for classification tasks
- SMOTE-like oversampling for minority classes during training
- Focal loss option for highly imbalanced classification

### 7. Population Binning for Classification

**Decision**: Use threshold of 20 FoB to create binary classes.

**Rationale**:
- Matches the approach in the original MSPB paper
- 20 FoB is a meaningful threshold for colony strength assessment
- Can extend to multi-class (low/medium/high) if needed

## Project Structure

```
hive-population-ml/
├── pyproject.toml           # uv project config
├── README.md                 # This file
├── src/
│   ├── data/
│   │   ├── loader.py        # Data loading utilities
│   │   ├── preprocessing.py # Feature engineering
│   │   └── dataset.py       # PyTorch Dataset classes
│   ├── models/
│   │   ├── mlp.py           # MLP baseline
│   │   ├── lstm.py          # LSTM/GRU model
│   │   ├── transformer.py   # Transformer model
│   │   └── multitask.py     # Multi-task prediction heads
│   ├── training/
│   │   ├── trainer.py       # Training loop
│   │   └── metrics.py       # Evaluation metrics
│   └── utils/
│       └── visualization.py # Plotting utilities
├── scripts/
│   ├── train.py             # Main training script
│   └── evaluate.py          # Evaluation script
└── configs/                  # Configuration files
```

## Installation

### Prerequisites
- Python 3.11+
- uv (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/spineda1208/hive-population-ml.git
cd hive-population-ml

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Data Setup

Download the MSPB dataset from [Zenodo](https://doi.org/10.5281/zenodo.8371700) and place the files in a `data/` directory:

```
data/
├── D1_sensor_data.csv
├── D2_sensor_data.csv
├── D1_ant.xlsx
└── D2_ant.xlsx
```

Or set the `MSPB_DATA_PATH` environment variable to point to the data location.

## Usage

### Training

```bash
# Train with default configuration (Transformer, all tasks)
uv run python scripts/train.py

# Train specific model
uv run python scripts/train.py --model mlp
uv run python scripts/train.py --model lstm
uv run python scripts/train.py --model transformer

# Train for specific task
uv run python scripts/train.py --task population
uv run python scripts/train.py --task winter_mortality

# Custom training parameters
uv run python scripts/train.py \
    --model transformer \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --window-days 7
```

### Inference

```bash
# Run inference on test set
uv run python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Run inference on new data
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data-path /path/to/new/sensor_data.csv
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | transformer | Model architecture (mlp, lstm, transformer) |
| `--task` | all | Prediction task (population, honey_yield, varroa, etc.) |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 32 | Training batch size |
| `--learning-rate` | 1e-4 | Initial learning rate |
| `--window-days` | 7 | Days of sensor data per sample |
| `--hidden-dim` | 256 | Hidden layer dimension |
| `--num-layers` | 3 | Number of model layers |
| `--dropout` | 0.1 | Dropout rate |
| `--data-path` | data/ | Path to MSPB data files |

## Results

Results will be logged to `results/` directory and include:
- Training curves (loss, metrics over epochs)
- Confusion matrices for classification tasks
- Scatter plots for regression tasks
- Model checkpoints

## Citation

If you use this code, please cite the original MSPB dataset paper:

```bibtex
@article{zhu2023mspb,
  title={MSPB: a longitudinal multi-sensor dataset with phenotypic trait measurements from honey bees (Apis mellifera L.)},
  author={Zhu, Yi and Abdollahi, Mahsa and Maucourt, S{\'e}gol{\`e}ne and others},
  journal={arXiv preprint arXiv:2311.10876},
  year={2023}
}
```

## License

MIT License
