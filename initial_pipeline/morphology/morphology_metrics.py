"""
Morphological Feature Similarity Metrics for Segmentation Evaluation

This module provides metrics to evaluate how similar predicted masks are to target masks
in terms of their morphological properties, rather than just pixel-level overlap.

Feature weights are loaded from analysis/shap_feature_weights.csv, which contains
SHAP-derived importance weights for all 25 morphological features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union


# ---------------------------------------------------------------------------
# All 25 morphological feature column names (must match get_morphology_dataframe())
# ---------------------------------------------------------------------------
ALL_FEATURE_COLUMNS = [
    'skeleton_length',
    'num_junctions',
    'num_components',
    'num_end_nodes',
    'num_start_nodes',
    'total_nodes',
    'end_to_start_ratio',
    'soma_area',
    'soma_perimeter',
    'soma_circularity',
    'cell_area',
    'cell_perimeter',
    'cell_convex_hull_area',
    'cell_convex_hull_perimeter',
    'cell_solidity',
    'cell_convexity',
    'cell_circularity',
    'cell_convex_circularity',
    'branch_area',
    'branch_perimeter',
    'sholl_min_radius',
    'sholl_peak_radius',
    'sholl_max_radius',
    'sholl_peak',
    'sholl_sum',
]

# Default path to SHAP weights CSV, relative to this file
_DEFAULT_SHAP_CSV = Path(__file__).parent.parent / "analysis" / "shap_feature_weights.csv"


def load_shap_weights(csv_path: Union[str, Path, None] = None) -> Dict[str, float]:
    """Load feature importance weights from a SHAP weights CSV file.

    The CSV must have columns 'feature' and 'weight'.  Weights are used as-is
    (they are already normalised to sum to ~1 by the SHAP calculation).

    Args:
        csv_path: Path to the CSV file.  If None, uses the default path
                  ``analysis/shap_feature_weights.csv`` relative to this file.

    Returns:
        Dictionary mapping feature name to SHAP weight.
    """
    if csv_path is None:
        csv_path = _DEFAULT_SHAP_CSV

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"SHAP weights CSV not found at {csv_path}. "
            "Pass an explicit csv_path or ensure the file exists."
        )

    df = pd.read_csv(csv_path)
    if 'feature' not in df.columns or 'weight' not in df.columns:
        raise ValueError(
            f"SHAP weights CSV at {csv_path} must have columns 'feature' and 'weight'. "
            f"Found: {list(df.columns)}"
        )

    return dict(zip(df['feature'], df['weight']))


def symmetric_relative_error(pred: float, target: float, epsilon: float = 1e-8) -> float:
    """Compute symmetric relative error: |pred - target| / (|pred| + |target| + epsilon).

    Returns a value in [0, 1] where 0 means perfect agreement.

    Args:
        pred: Predicted value
        target: Target value
        epsilon: Small constant to avoid division by zero

    Returns:
        Symmetric relative error in [0, 1], where 0 is perfect
    """
    return abs(pred - target) / (abs(pred) + abs(target) + epsilon)


def per_feature_morphology_score(pred_features: pd.Series, target_features: pd.Series,
                                  weights: Optional[Union[Dict[str, float], None]] = None,
                                  epsilon: float = 1e-8) -> Dict[str, Dict[str, float]]:
    """Compute a detailed per-feature breakdown of the morphology score.

    Performance for each feature is defined as ``1 - symmetric_relative_error``
    on the raw feature values, giving an intuitive accuracy in [0, 1] where
    1.0 = perfect match and 0.0 = maximally wrong.

    Args:
        pred_features: Predicted morphological features (pandas Series).
        target_features: Target morphological features (pandas Series).
        weights: Feature weights as a dict mapping feature name → weight.
                 If None, SHAP weights are loaded from the default CSV path.
        epsilon: Small constant to avoid division by zero.

    Returns:
        Dictionary mapping each feature name to a dict with keys:
            ``error``        – symmetric relative error in [0, 1]  (0 = perfect)
            ``performance``  – 1 - error, i.e. % accuracy in [0, 1]  (1 = perfect)
            ``weight``       – SHAP importance weight
            ``contribution`` – weight × performance (un-normalised contribution)
    """
    if weights is None:
        weights = load_shap_weights()

    result: Dict[str, Dict[str, float]] = {}
    for feature in pred_features.index:
        if feature not in target_features.index:
            continue
        error = symmetric_relative_error(
            pred_features[feature], target_features[feature], epsilon
        )
        performance = 1.0 - error
        weight = weights.get(feature, 0.0)
        result[feature] = {
            "error": error,
            "performance": performance,
            "weight": weight,
            "contribution": weight * performance,
        }

    return result


def weighted_morphology_score(pred_features: pd.Series, target_features: pd.Series,
                               weights: Optional[Union[Dict[str, float], None]] = None,
                               epsilon: float = 1e-8) -> float:
    """Compute weighted average morphology score as a weighted mean of per-feature accuracy.

    Each feature's accuracy is ``1 - symmetric_relative_error(pred, target)``.
    The final score is the SHAP-weighted average of those per-feature accuracies.

    Args:
        pred_features: Predicted morphological features
        target_features: Target morphological features
        weights: Feature weights as a dict mapping feature name → weight.
                 If None, SHAP weights are loaded from the default CSV path.
        epsilon: Small constant to avoid division by zero

    Returns:
        Weighted morphology score in [0, 1], where 1 is perfect
    """
    breakdown = per_feature_morphology_score(pred_features, target_features, weights, epsilon)

    total_weight = 0.0
    weighted_sum = 0.0

    for feature, vals in breakdown.items():
        weighted_sum += vals["contribution"]
        total_weight += vals["weight"]

    if total_weight > 0:
        return weighted_sum / total_weight
    else:
        return 0.0
