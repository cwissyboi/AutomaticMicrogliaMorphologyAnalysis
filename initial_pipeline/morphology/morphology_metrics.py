"""
Morphological Feature Similarity Metrics for Segmentation Evaluation

This module provides metrics to evaluate how similar predicted masks are to target masks
in terms of their morphological properties, rather than just pixel-level overlap.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class MorphologyMetricWeights:
    """Weights for different morphological features in composite metrics.
    
    Default weights prioritize features that are:
    - Most relevant for microglia phenotype classification
    - Most sensitive to segmentation quality
    """
    # Skeleton-based features (topology)
    length_pixels: float = 1.0
    num_branches: float = 2.0  # Higher weight - critical for phenotype
    num_components: float = 1.5  # Higher weight - fragmentation is important
    
    # Soma features
    soma_area: float = 0.8
    soma_perimeter: float = 0.5
    soma_circularity: float = 0.7
    
    # Cell shape features
    cell_area: float = 1.0
    cell_perimeter: float = 0.6
    cell_convex_hull_area: float = 0.7
    cell_convex_hull_perimeter: float = 0.5
    cell_solidity: float = 1.2  # Higher weight - good phenotype indicator
    cell_convexity: float = 1.2  # Higher weight - good phenotype indicator
    cell_circularity: float = 1.0
    
    def get_weights_dict(self) -> Dict[str, float]:
        """Return weights as a dictionary."""
        return {
            'length_pixels': self.length_pixels,
            'num_branches': self.num_branches,
            'num_components': self.num_components,
            'soma_area': self.soma_area,
            'soma_perimeter': self.soma_perimeter,
            'soma_circularity': self.soma_circularity,
            'cell_area': self.cell_area,
            'cell_perimeter': self.cell_perimeter,
            'cell_convex_hull_area': self.cell_convex_hull_area,
            'cell_convex_hull_perimeter': self.cell_convex_hull_perimeter,
            'cell_solidity': self.cell_solidity,
            'cell_convexity': self.cell_convexity,
            'cell_circularity': self.cell_circularity,
        }


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
        # Compute ranges from data
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
        
        # Avoid division by zero
        if max_val - min_val > 1e-8:
            pred_norm[feature] = (pred_features[feature] - min_val) / (max_val - min_val)
            target_norm[feature] = (target_features[feature] - min_val) / (max_val - min_val)
        else:
            pred_norm[feature] = 0.0
            target_norm[feature] = 0.0
    
    return pred_norm, target_norm


def relative_error(pred: float, target: float, epsilon: float = 1e-8) -> float:
    """Compute relative error: |pred - target| / (target + epsilon).
    
    Args:
        pred: Predicted value
        target: Target value
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Relative error in [0, inf), where 0 is perfect
    """
    return abs(pred - target) / (abs(target) + epsilon)


def symmetric_relative_error(pred: float, target: float, epsilon: float = 1e-8) -> float:
    """Compute symmetric relative error: |pred - target| / (pred + target + epsilon).
    
    More balanced than relative_error when pred can be larger than target.
    
    Args:
        pred: Predicted value
        target: Target value
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Symmetric relative error in [0, 1], where 0 is perfect
    """
    return abs(pred - target) / (abs(pred) + abs(target) + epsilon)


def morphology_similarity_score(pred: float, target: float, epsilon: float = 1e-8) -> float:
    """Convert error to similarity score in [0, 1], where 1 is perfect.
    
    Uses symmetric relative error converted to similarity.
    
    Args:
        pred: Predicted value
        target: Target value
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Similarity score in [0, 1], where 1 is perfect match
    """
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
            similarities[feature] = morphology_similarity_score(
                pred_norm[feature], target_norm[feature], epsilon
            )
    
    return similarities


def weighted_morphology_score(pred_features: pd.Series, target_features: pd.Series,
                               weights: Optional[Union[Dict[str, float], MorphologyMetricWeights]] = None,
                               normalize: bool = True,
                               epsilon: float = 1e-8) -> float:
    """Compute weighted average of morphological feature similarities.
    
    This is the main metric for overall morphological similarity.
    
    Args:
        pred_features: Predicted morphological features
        target_features: Target morphological features
        weights: Feature weights (dict or MorphologyMetricWeights).
                If None, uses default weights.
        normalize: Whether to normalize features before comparison
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Weighted morphology score in [0, 1], where 1 is perfect
    """
    if weights is None:
        weights = MorphologyMetricWeights()
    
    if isinstance(weights, MorphologyMetricWeights):
        weights = weights.get_weights_dict()
    
    # Get per-feature similarities
    similarities = per_feature_similarity(pred_features, target_features, normalize, epsilon)
    
    # Compute weighted average
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


def phenotype_critical_score(pred_features: pd.Series, target_features: pd.Series,
                              normalize: bool = True,
                              epsilon: float = 1e-8) -> float:
    """Compute similarity score focusing on phenotype-critical features.
    
    Focuses on: num_branches, num_components, cell_solidity, cell_convexity
    These features are most important for distinguishing microglia phenotypes
    (ramified vs. amoeboid).
    
    Args:
        pred_features: Predicted morphological features
        target_features: Target morphological features
        normalize: Whether to normalize features before comparison
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Phenotype-critical score in [0, 1], where 1 is perfect
    """
    phenotype_weights = {
        'num_branches': 1.0,
        'num_components': 1.0,
        'cell_solidity': 1.0,
        'cell_convexity': 1.0,
    }
    
    return weighted_morphology_score(pred_features, target_features,
                                    weights=phenotype_weights,
                                    normalize=normalize,
                                    epsilon=epsilon)


def topology_score(pred_features: pd.Series, target_features: pd.Series,
                   normalize: bool = True,
                   epsilon: float = 1e-8) -> float:
    """Compute similarity score focusing on topological features.
    
    Focuses on: length_pixels, num_branches, num_components
    These features capture the skeleton topology.
    
    Args:
        pred_features: Predicted morphological features
        target_features: Target morphological features
        normalize: Whether to normalize features before comparison
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Topology score in [0, 1], where 1 is perfect
    """
    topology_weights = {
        'length_pixels': 1.0,
        'num_branches': 1.0,
        'num_components': 1.0,
    }
    
    return weighted_morphology_score(pred_features, target_features,
                                    weights=topology_weights,
                                    normalize=normalize,
                                    epsilon=epsilon)


def shape_score(pred_features: pd.Series, target_features: pd.Series,
                normalize: bool = True,
                epsilon: float = 1e-8) -> float:
    """Compute similarity score focusing on shape features.
    
    Focuses on: cell_solidity, cell_convexity, cell_circularity
    These features capture the overall shape characteristics.
    
    Args:
        pred_features: Predicted morphological features
        target_features: Target morphological features
        normalize: Whether to normalize features before comparison
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Shape score in [0, 1], where 1 is perfect
    """
    shape_weights = {
        'cell_solidity': 1.0,
        'cell_convexity': 1.0,
        'cell_circularity': 1.0,
        'soma_circularity': 1.0,
    }
    
    return weighted_morphology_score(pred_features, target_features,
                                    weights=shape_weights,
                                    normalize=normalize,
                                    epsilon=epsilon)


def evaluate_morphology_batch(pred_features_df: pd.DataFrame,
                               target_features_df: pd.DataFrame,
                               cell_id_column: str = 'cell_id',
                               weights: Optional[Union[Dict[str, float], MorphologyMetricWeights]] = None,
                               normalize: bool = True) -> Dict[str, float]:
    """Evaluate morphological similarity for a batch of cells (e.g., test set).
    
    Args:
        pred_features_df: DataFrame with predicted features for multiple cells
        target_features_df: DataFrame with target features for multiple cells
        cell_id_column: Column name to match cells between pred and target
        weights: Feature weights for weighted score
        normalize: Whether to normalize features before comparison
    
    Returns:
        Dictionary with mean scores across all cells:
        - 'weighted_morphology_score': Overall weighted similarity
        - 'phenotype_critical_score': Phenotype-critical features similarity
        - 'topology_score': Skeleton topology similarity
        - 'shape_score': Shape features similarity
        - 'per_feature_scores': Dict of mean similarity for each feature
        - 'num_cells': Number of cells evaluated
    """
    feature_columns = [
        'length_pixels', 'num_branches', 'num_components',
        'soma_area', 'soma_perimeter', 'soma_circularity',
        'cell_area', 'cell_perimeter', 'cell_convex_hull_area',
        'cell_convex_hull_perimeter', 'cell_solidity', 'cell_convexity',
        'cell_circularity'
    ]
    
    # Filter to only include feature columns that exist in both DataFrames
    feature_columns = [col for col in feature_columns 
                      if col in pred_features_df.columns and col in target_features_df.columns]
    
    # Match cells by ID
    if cell_id_column in pred_features_df.columns and cell_id_column in target_features_df.columns:
        # Merge on cell_id to align predictions with targets
        merged = pred_features_df.merge(target_features_df, on=cell_id_column, 
                                       suffixes=('_pred', '_target'))
        
        pred_cols = [f"{col}_pred" for col in feature_columns]
        target_cols = [f"{col}_target" for col in feature_columns]
    else:
        # Assume same order if no ID column
        merged = pd.concat([pred_features_df.reset_index(drop=True),
                          target_features_df.reset_index(drop=True)],
                         axis=1, keys=['pred', 'target'])
        pred_cols = [('pred', col) for col in feature_columns]
        target_cols = [('target', col) for col in feature_columns]
    
    num_cells = len(merged)
    
    # Initialize score accumulators
    weighted_scores = []
    phenotype_scores = []
    topology_scores = []
    shape_scores = []
    per_feature_scores = {col: [] for col in feature_columns}
    
    # Compute scores for each cell
    for idx, row in merged.iterrows():
        if cell_id_column in pred_features_df.columns:
            pred_feats = pd.Series({col: row[f"{col}_pred"] for col in feature_columns})
            target_feats = pd.Series({col: row[f"{col}_target"] for col in feature_columns})
        else:
            pred_feats = pd.Series({col: row[('pred', col)] for col in feature_columns})
            target_feats = pd.Series({col: row[('target', col)] for col in feature_columns})
        
        # Weighted morphology score
        weighted_scores.append(
            weighted_morphology_score(pred_feats, target_feats, weights, normalize)
        )
        
        # Specialized scores
        phenotype_scores.append(
            phenotype_critical_score(pred_feats, target_feats, normalize)
        )
        topology_scores.append(
            topology_score(pred_feats, target_feats, normalize)
        )
        shape_scores.append(
            shape_score(pred_feats, target_feats, normalize)
        )
        
        # Per-feature scores
        feature_sims = per_feature_similarity(pred_feats, target_feats, normalize)
        for feature, score in feature_sims.items():
            per_feature_scores[feature].append(score)
    
    # Compute means
    results = {
        'weighted_morphology_score': np.mean(weighted_scores),
        'phenotype_critical_score': np.mean(phenotype_scores),
        'topology_score': np.mean(topology_scores),
        'shape_score': np.mean(shape_scores),
        'per_feature_scores': {
            feature: np.mean(scores) for feature, scores in per_feature_scores.items()
            if len(scores) > 0
        },
        'num_cells': num_cells,
    }
    
    return results


def print_morphology_evaluation(results: Dict[str, float], verbose: bool = True):
    """Pretty print morphology evaluation results.
    
    Args:
        results: Output from evaluate_morphology_batch()
        verbose: If True, print per-feature scores
    """
    print("\n" + "="*60)
    print("MORPHOLOGICAL SIMILARITY EVALUATION")
    print("="*60)
    print(f"Number of cells evaluated: {results['num_cells']}")
    print("\n" + "-"*60)
    print("AGGREGATE SCORES (0=worst, 1=best)")
    print("-"*60)
    print(f"  Weighted Morphology Score:    {results['weighted_morphology_score']:.4f}")
    print(f"  Phenotype-Critical Score:     {results['phenotype_critical_score']:.4f}")
    print(f"  Topology Score:               {results['topology_score']:.4f}")
    print(f"  Shape Score:                  {results['shape_score']:.4f}")
    
    if verbose and 'per_feature_scores' in results:
        print("\n" + "-"*60)
        print("PER-FEATURE SIMILARITY SCORES")
        print("-"*60)
        for feature, score in sorted(results['per_feature_scores'].items()):
            print(f"  {feature:30s}: {score:.4f}")
    
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Example: single cell comparison
    pred = pd.Series({
        'length_pixels': 450,
        'num_branches': 12,
        'num_components': 1,
        'soma_area': 850,
        'soma_perimeter': 110,
        'soma_circularity': 0.88,
        'cell_area': 2100,
        'cell_perimeter': 320,
        'cell_convex_hull_area': 2400,
        'cell_convex_hull_perimeter': 280,
        'cell_solidity': 0.875,
        'cell_convexity': 0.875,
        'cell_circularity': 0.65,
    })
    
    target = pd.Series({
        'length_pixels': 500,
        'num_branches': 15,
        'num_components': 1,
        'soma_area': 900,
        'soma_perimeter': 120,
        'soma_circularity': 0.78,
        'cell_area': 2200,
        'cell_perimeter': 340,
        'cell_convex_hull_area': 2500,
        'cell_convex_hull_perimeter': 290,
        'cell_solidity': 0.88,
        'cell_convexity': 0.853,
        'cell_circularity': 0.60,
    })
    
    print("Single Cell Example:")
    print(f"Weighted Morphology Score: {weighted_morphology_score(pred, target):.4f}")
    print(f"Phenotype Critical Score: {phenotype_critical_score(pred, target):.4f}")
    print(f"Topology Score: {topology_score(pred, target):.4f}")
    print(f"Shape Score: {shape_score(pred, target):.4f}")
    print("\nPer-feature similarities:")
    for feature, score in per_feature_similarity(pred, target).items():
        print(f"  {feature}: {score:.4f}")
