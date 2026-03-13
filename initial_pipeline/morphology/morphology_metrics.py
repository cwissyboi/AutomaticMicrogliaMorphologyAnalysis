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
from typing import Dict, Tuple, Optional, Union


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


def normalize_features(pred_features: pd.Series, target_features: pd.Series,
                       feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
                       ) -> Tuple[pd.Series, pd.Series]:
    """Normalize features to [0, 1] range for fair comparison.

    Args:
        pred_features: Predicted morphological features
        target_features: Target morphological features
        feature_ranges: Optional dict of (min, max) for each feature.
                       If None, uses min/max from pred and target.

    Returns:
        Normalized (pred_features, target_features)
    """
    if feature_ranges is None:
        feature_ranges = {}
        for feature in pred_features.index:
            min_val = min(pred_features[feature], target_features[feature])
            max_val = max(pred_features[feature], target_features[feature])
            feature_ranges[feature] = (min_val, max_val)

    pred_norm = pred_features.copy()
    target_norm = target_features.copy()

    for feature, (min_val, max_val) in feature_ranges.items():
        if feature not in pred_features.index:
            continue

        if max_val - min_val > 1e-8:
            pred_norm[feature] = (pred_features[feature] - min_val) / (max_val - min_val)
            target_norm[feature] = (target_features[feature] - min_val) / (max_val - min_val)
        else:
            pred_norm[feature] = 0.0
            target_norm[feature] = 0.0

    return pred_norm, target_norm


def symmetric_relative_error(pred: float, target: float, epsilon: float = 1e-8) -> float:
    """Compute symmetric relative error: |pred - target| / (pred + target + epsilon).

    More balanced than one-sided relative error when pred can be larger than target.

    Args:
        pred: Predicted value
        target: Target value
        epsilon: Small constant to avoid division by zero

    Returns:
        Symmetric relative error in [0, 1], where 0 is perfect
    """
    return abs(pred - target) / (abs(pred) + abs(target) + epsilon)


def _feature_similarity(pred: float, target: float, epsilon: float = 1e-8) -> float:
    """Convert symmetric relative error to a similarity score in [0, 1]."""
    error = symmetric_relative_error(pred, target, epsilon)
    return 1.0 / (1.0 + error)


def per_feature_similarity(pred_features: pd.Series, target_features: pd.Series,
                            normalize: bool = True,
                            epsilon: float = 1e-8) -> Dict[str, float]:
    """Compute similarity scores for each morphological feature independently.

    Args:
        pred_features: Predicted morphological features (pandas Series)
        target_features: Target morphological features (pandas Series)
        normalize: Whether to normalize features before comparison
        epsilon: Small constant to avoid division by zero

    Returns:
        Dictionary mapping feature name to similarity score [0, 1]
    """
    if normalize:
        pred_norm, target_norm = normalize_features(pred_features, target_features)
    else:
        pred_norm, target_norm = pred_features, target_features

    similarities = {}
    for feature in pred_norm.index:
        if feature in target_norm.index:
            similarities[feature] = _feature_similarity(
                pred_norm[feature], target_norm[feature], epsilon
            )

    return similarities


def weighted_morphology_score(pred_features: pd.Series, target_features: pd.Series,
                               weights: Optional[Union[Dict[str, float], None]] = None,
                               normalize: bool = True,
                               epsilon: float = 1e-8) -> float:
    """Compute weighted average of morphological feature similarities.

    This is the main metric for overall morphological similarity.  When no
    weights are provided the SHAP-derived weights are loaded automatically
    from ``analysis/shap_feature_weights.csv``.

    Args:
        pred_features: Predicted morphological features
        target_features: Target morphological features
        weights: Feature weights as a dict mapping feature name → weight.
                 If None, SHAP weights are loaded from the default CSV path.
        normalize: Whether to normalize features before comparison
        epsilon: Small constant to avoid division by zero

    Returns:
        Weighted morphology score in [0, 1], where 1 is perfect
    """
    if weights is None:
        weights = load_shap_weights()

    similarities = per_feature_similarity(pred_features, target_features, normalize, epsilon)

    total_weight = 0.0
    weighted_sum = 0.0

    for feature, similarity in similarities.items():
        if feature in weights:
            weight = weights[feature]
            weighted_sum += weight * similarity
            total_weight += weight

    if total_weight > 0:
        return weighted_sum / total_weight
    else:
        return 0.0
