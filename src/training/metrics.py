"""
Evaluation metrics for hive prediction tasks.

Supports both regression and classification metrics.
"""

from typing import Dict, Optional, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "binary",
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)
        average: Averaging method for multi-class

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    # Handle binary vs multi-class
    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        avg = "binary"
    else:
        avg = average

    metrics["f1"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=avg, zero_division=0)

    # AUC if probabilities provided
    if y_prob is not None:
        try:
            if n_classes == 2:
                # Binary case - use probability of positive class
                if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                    metrics["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
            else:
                # Multi-class case
                metrics["auc_roc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            # May fail if only one class present
            metrics["auc_roc"] = 0.0

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["mse"] = mean_squared_error(y_true, y_pred)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["r2"] = r2_score(y_true, y_pred)

    # Normalized metrics
    y_range = y_true.max() - y_true.min()
    if y_range > 0:
        metrics["nrmse"] = metrics["rmse"] / y_range
    else:
        metrics["nrmse"] = 0.0

    # Mean absolute percentage error (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        metrics["mape"] = (
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        )
    else:
        metrics["mape"] = 0.0

    return metrics


def get_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None
) -> Dict:
    """
    Get confusion matrix and related statistics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional class labels

    Returns:
        Dictionary with confusion matrix and per-class metrics
    """
    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    n_classes = cm.shape[0]
    per_class = {}

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        class_label = labels[i] if labels and i < len(labels) else f"class_{i}"
        per_class[class_label] = {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "support": int(cm[i, :].sum()),
        }

    return {"confusion_matrix": cm.tolist(), "per_class": per_class}


class MetricTracker:
    """
    Tracks and accumulates metrics over batches/epochs.
    """

    def __init__(self, task_type: str = "classification"):
        """
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.reset()

    def reset(self):
        """Reset all accumulated values."""
        self.y_true_list = []
        self.y_pred_list = []
        self.y_prob_list = []
        self.loss_sum = 0.0
        self.n_samples = 0

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        loss: float = 0.0,
        n_samples: int = 1,
    ):
        """
        Add batch results.

        Args:
            y_true: Ground truth values/labels
            y_pred: Predictions
            y_prob: Predicted probabilities (classification only)
            loss: Batch loss value
            n_samples: Number of samples in batch
        """
        self.y_true_list.append(y_true)
        self.y_pred_list.append(y_pred)
        if y_prob is not None:
            self.y_prob_list.append(y_prob)
        self.loss_sum += loss * n_samples
        self.n_samples += n_samples

    def compute(self) -> Dict[str, float]:
        """
        Compute metrics from accumulated values.

        Returns:
            Dictionary of metric names to values
        """
        if self.n_samples == 0:
            return {}

        y_true = np.concatenate(self.y_true_list)
        y_pred = np.concatenate(self.y_pred_list)
        y_prob = np.concatenate(self.y_prob_list) if self.y_prob_list else None

        # Compute metrics
        if self.task_type == "classification":
            metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        else:
            metrics = compute_regression_metrics(y_true, y_pred)

        # Add loss
        metrics["loss"] = self.loss_sum / self.n_samples

        return metrics

    def get_confusion_matrix(self, labels: Optional[List[str]] = None) -> Dict:
        """Get confusion matrix (classification only)."""
        if self.task_type != "classification":
            return {}

        y_true = np.concatenate(self.y_true_list)
        y_pred = np.concatenate(self.y_pred_list)

        return get_confusion_matrix(y_true, y_pred, labels)


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary as a readable string.

    Args:
        metrics: Dictionary of metric names to values
        precision: Decimal places for formatting

    Returns:
        Formatted string
    """
    parts = []
    for name, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{name}: {value:.{precision}f}")
        else:
            parts.append(f"{name}: {value}")
    return " | ".join(parts)


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)

    # Classification test
    print("Testing classification metrics...")
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_prob = np.random.rand(8, 2)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    print(format_metrics(metrics))

    cm_result = get_confusion_matrix(y_true, y_pred, ["Low", "High"])
    print(f"Confusion matrix: {cm_result['confusion_matrix']}")

    # Regression test
    print("\nTesting regression metrics...")
    y_true = np.array([15.0, 20.0, 25.0, 30.0, 35.0])
    y_pred = np.array([14.0, 22.0, 24.0, 28.0, 36.0])

    metrics = compute_regression_metrics(y_true, y_pred)
    print(format_metrics(metrics))

    # Test MetricTracker
    print("\nTesting MetricTracker...")
    tracker = MetricTracker("classification")

    for i in range(3):
        batch_true = np.random.randint(0, 2, 8)
        batch_pred = np.random.randint(0, 2, 8)
        tracker.update(batch_true, batch_pred, loss=0.5 + i * 0.1, n_samples=8)

    final_metrics = tracker.compute()
    print(format_metrics(final_metrics))
