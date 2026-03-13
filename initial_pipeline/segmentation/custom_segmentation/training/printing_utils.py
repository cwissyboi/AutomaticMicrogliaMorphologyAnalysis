"""
Utilities for printing cross-validation results in a formatted way.

This module provides functions to print standard metrics, region-based metrics,
and comparison between postprocessed and non-postprocessed results.
"""

import numpy as np
from typing import List, Dict, Optional


def print_morphology_feature_summary(
    all_fold_feature_breakdowns: List[List[Dict[str, Dict[str, float]]]],
    label: Optional[str] = None,
):
    """Print a per-feature morphology summary averaged across all folds.

    Each fold contributes a list of per-cell breakdowns produced by
    ``morphology_similarity_score_detailed``.  This function aggregates those
    across both folds and cells, then prints a table showing for every feature:

    * **Weight**       – SHAP importance weight (same for all cells)
    * **Performance**  – mean similarity × 100  (i.e. % performance; 100% = perfect)
    * **Error**        – mean symmetric relative error (0 = perfect, 1 = worst)
    * **Contribution** – mean (weight × similarity) / total_weight × 100
                         showing how much each feature contributes to the final score

    Args:
        all_fold_feature_breakdowns: Outer list = one entry per fold; each entry
            is a list of per-cell breakdown dicts as returned by
            ``morphology_similarity_score_detailed``.
        label: Optional display label (e.g. ``"NO POSTPROCESSING"``).
    """
    # Flatten all per-cell breakdowns across folds into a single list
    all_cell_breakdowns: List[Dict[str, Dict[str, float]]] = []
    for fold_breakdowns in all_fold_feature_breakdowns:
        all_cell_breakdowns.extend(fold_breakdowns)

    if not all_cell_breakdowns:
        print("\nNo morphology feature breakdown data available.")
        return

    # Collect feature names (assume all cells share the same feature set)
    feature_names = list(all_cell_breakdowns[0].keys())

    # Accumulate per-feature statistics
    feature_stats: Dict[str, Dict[str, List[float]]] = {
        f: {"similarity": [], "error": [], "weight": [], "contribution": []}
        for f in feature_names
    }
    for cell_breakdown in all_cell_breakdowns:
        for feature, vals in cell_breakdown.items():
            if feature in feature_stats:
                feature_stats[feature]["similarity"].append(vals["similarity"])
                feature_stats[feature]["error"].append(vals["error"])
                feature_stats[feature]["weight"].append(vals["weight"])
                feature_stats[feature]["contribution"].append(vals["contribution"])

    # Compute mean values
    mean_stats = {}
    total_weight = 0.0
    for feature in feature_names:
        s = feature_stats[feature]
        mean_weight = float(np.mean(s["weight"])) if s["weight"] else 0.0
        mean_stats[feature] = {
            "similarity": float(np.mean(s["similarity"])) if s["similarity"] else 0.0,
            "error":      float(np.mean(s["error"]))      if s["error"]      else 0.0,
            "weight":     mean_weight,
            "contribution": float(np.mean(s["contribution"])) if s["contribution"] else 0.0,
        }
        total_weight += mean_weight

    # Weighted score from this summary (should match reported morphology score)
    weighted_score = (
        sum(v["contribution"] for v in mean_stats.values()) / total_weight
        if total_weight > 0 else 0.0
    )

    # Sort features by weight descending (highest-impact first)
    sorted_features = sorted(feature_names, key=lambda f: mean_stats[f]["weight"], reverse=True)

    title = f"MORPHOLOGY FEATURE SUMMARY"
    if label:
        title += f" ({label})"

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total Morphology Score : {weighted_score * 100:.2f}%")
    print(f"Cells evaluated        : {len(all_cell_breakdowns)}")
    print()

    # Column widths
    col_feature = 28
    col_weight  = 8
    col_perf    = 13
    col_err     = 10
    col_contrib = 14

    header = (
        f"{'Feature':<{col_feature}}"
        f"{'Weight':>{col_weight}}"
        f"{'Performance':>{col_perf}}"
        f"{'Error':>{col_err}}"
        f"{'Contribution':>{col_contrib}}"
    )
    sep = "-" * 80
    print(header)
    print(sep)

    for feature in sorted_features:
        v = mean_stats[feature]
        weight_pct  = v["weight"]  / total_weight * 100 if total_weight > 0 else 0.0
        perf_pct    = v["similarity"] * 100
        error_val   = v["error"]
        # Contribution as % of the final score denominator
        contrib_pct = v["contribution"] / total_weight * 100 if total_weight > 0 else 0.0

        print(
            f"{feature:<{col_feature}}"
            f"{weight_pct:>{col_weight}.2f}%"
            f"{perf_pct:>{col_perf - 1}.2f}%"
            f"{error_val:>{col_err}.4f}"
            f"{contrib_pct:>{col_contrib - 1}.2f}%"
        )

    print(sep)
    print(
        f"{'TOTAL':<{col_feature}}"
        f"{'100.00%':>{col_weight + 1}}"
        f"{'':>{col_perf}}"
        f"{'':>{col_err}}"
        f"{weighted_score * 100:>{col_contrib - 1}.2f}%"
    )
    print("=" * 80)
    print("  Weight:       % share of total SHAP importance")
    print("  Performance:  mean per-feature similarity (100% = perfect match)")
    print("  Error:        mean symmetric relative error (0 = perfect)")
    print("  Contribution: weighted contribution to the final morphology score")
    print("=" * 80 + "\n")


def print_standard_metrics(
    all_fold_metrics: List[Dict],
    with_postprocessing: bool = False,
    label: Optional[str] = None
):
    """
    Print standard metrics (Dice, IoU, Morphology) across all folds.
    
    Args:
        all_fold_metrics: List of dictionaries containing metrics from each fold
        with_postprocessing: Whether these results are from postprocessed predictions
        label: Optional custom label (e.g., "ADAPTIVE", "PROBABILITY"). If not provided,
               uses "WITH POSTPROCESSING" or "NO POSTPROCESSING" based on with_postprocessing flag.
    """
    dice_scores = [m["dice"] for m in all_fold_metrics]
    iou_scores = [m["iou"] for m in all_fold_metrics]
    morphology_scores = [m["morphology"] for m in all_fold_metrics]
    
    if label:
        display_label = label
    else:
        display_label = "WITH POSTPROCESSING" if with_postprocessing else "NO POSTPROCESSING"
    
    print("\n" + "="*70)
    print(f"CROSS-VALIDATION RESULTS (Standard Metrics - {display_label})")
    print("="*70)
    print(f"Dice:       {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"IoU:        {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"Morphology: {np.mean(morphology_scores):.4f} ± {np.std(morphology_scores):.4f}")


def print_standard_metrics_comparison(
    baseline_metrics: List[Dict],
    postprocessed_metrics: List[Dict]
):
    """
    Print comparison between baseline and postprocessed standard metrics.
    
    Args:
        baseline_metrics: Metrics without postprocessing
        postprocessed_metrics: Metrics with postprocessing
    """
    baseline_dice = [m["dice"] for m in baseline_metrics]
    baseline_iou = [m["iou"] for m in baseline_metrics]
    baseline_morphology = [m["morphology"] for m in baseline_metrics]
    
    pp_dice = [m["dice"] for m in postprocessed_metrics]
    pp_iou = [m["iou"] for m in postprocessed_metrics]
    pp_morphology = [m["morphology"] for m in postprocessed_metrics]
    
    print("\nCOMPARISON (Postprocessed vs No Postprocessing):")
    print(f"  Dice improvement:       {np.mean(pp_dice) - np.mean(baseline_dice):+.4f}")
    print(f"  IoU improvement:        {np.mean(pp_iou) - np.mean(baseline_iou):+.4f}")
    print(f"  Morphology improvement: {np.mean(pp_morphology) - np.mean(baseline_morphology):+.4f}")


def print_region_based_overall_metrics(all_fold_region_metrics: List[Dict]):
    """
    Print overall (entire cell) region-based metrics.
    
    Args:
        all_fold_region_metrics: List of region-based metrics from each fold
    """
    overall_dice = [fold['overall']['dice_mean'] for fold in all_fold_region_metrics]
    overall_iou = [fold['overall']['iou_mean'] for fold in all_fold_region_metrics]
    overall_precision = [fold['overall']['precision_mean'] for fold in all_fold_region_metrics]
    overall_recall = [fold['overall']['recall_mean'] for fold in all_fold_region_metrics]
    
    print("OVERALL (Entire Cell):")
    print(f"  Dice:      {np.mean(overall_dice):.4f} ± {np.std(overall_dice):.4f}")
    print(f"  IoU:       {np.mean(overall_iou):.4f} ± {np.std(overall_iou):.4f}")
    print(f"  Precision: {np.mean(overall_precision):.4f} ± {np.std(overall_precision):.4f}")
    print(f"  Recall:    {np.mean(overall_recall):.4f} ± {np.std(overall_recall):.4f}")


def print_region_based_soma_metrics(all_fold_region_metrics: List[Dict]):
    """
    Print soma-specific region-based metrics (recall only).
    
    Args:
        all_fold_region_metrics: List of region-based metrics from each fold
    """
    if not all('soma' in fold for fold in all_fold_region_metrics):
        return
    
    soma_recall = [fold['soma']['recall_mean'] for fold in all_fold_region_metrics]
    soma_sample_count = [fold['soma']['sample_count'] for fold in all_fold_region_metrics]
    soma_pixel_count = [fold['soma']['avg_pixel_count'] for fold in all_fold_region_metrics]
    
    print("\nSOMA (Cell Body) - RECALL ONLY:")
    print(f"  Avg Samples per Fold: {np.mean(soma_sample_count):.1f}")
    print(f"  Avg Pixels:           {np.mean(soma_pixel_count):.0f}")
    print(f"  Recall:               {np.mean(soma_recall):.4f} ± {np.std(soma_recall):.4f}")
    print(f"  (Proportion of GT soma pixels captured by prediction)")


def print_region_based_branches_metrics(all_fold_region_metrics: List[Dict]):
    """
    Print branches-specific region-based metrics.
    
    Args:
        all_fold_region_metrics: List of region-based metrics from each fold
    """
    if not all('branches' in fold for fold in all_fold_region_metrics):
        return
    
    branches_dice = [fold['branches']['dice_mean'] for fold in all_fold_region_metrics]
    branches_iou = [fold['branches']['iou_mean'] for fold in all_fold_region_metrics]
    branches_precision = [fold['branches']['precision_mean'] for fold in all_fold_region_metrics]
    branches_recall = [fold['branches']['recall_mean'] for fold in all_fold_region_metrics]
    branches_sample_count = [fold['branches']['sample_count'] for fold in all_fold_region_metrics]
    branches_pixel_count = [fold['branches']['avg_pixel_count'] for fold in all_fold_region_metrics]
    
    print("\nBRANCHES (Arms/Processes) - FULL METRICS:")
    print(f"  Avg Samples per Fold: {np.mean(branches_sample_count):.1f}")
    print(f"  Avg Pixels:           {np.mean(branches_pixel_count):.0f}")
    print(f"  Dice:                 {np.mean(branches_dice):.4f} ± {np.std(branches_dice):.4f}")
    print(f"  IoU:                  {np.mean(branches_iou):.4f} ± {np.std(branches_iou):.4f}")
    print(f"  Precision:            {np.mean(branches_precision):.4f} ± {np.std(branches_precision):.4f}")
    print(f"  Recall:               {np.mean(branches_recall):.4f} ± {np.std(branches_recall):.4f}")
    print(f"  (Predictions outside soma region vs GT branches)")


def print_region_based_metrics(
    all_fold_region_metrics: List[Dict],
    with_postprocessing: bool = False
):
    """
    Print complete region-based metrics (overall, soma, branches).
    
    Args:
        all_fold_region_metrics: List of region-based metrics from each fold
        with_postprocessing: Whether these results are from postprocessed predictions
    """
    if not all_fold_region_metrics:
        print("\nNo region-based metrics were computed (no soma masks found in any fold)")
        return
    
    label = "WITH POSTPROCESSING" if with_postprocessing else "NO POSTPROCESSING"
    print("\n" + "="*70)
    print(f"CROSS-VALIDATION RESULTS (Region-Based Metrics - {label})")
    print("="*70)
    print(f"Based on {len(all_fold_region_metrics)} folds with region annotations\n")
    
    print_region_based_overall_metrics(all_fold_region_metrics)
    print_region_based_soma_metrics(all_fold_region_metrics)
    print_region_based_branches_metrics(all_fold_region_metrics)
    
    print("="*70 + "\n")


def print_region_based_overall_comparison(
    baseline_metrics: List[Dict],
    postprocessed_metrics: List[Dict]
):
    """
    Print comparison of overall region-based metrics.
    
    Args:
        baseline_metrics: Metrics without postprocessing
        postprocessed_metrics: Metrics with postprocessing
    """
    baseline_dice = [fold['overall']['dice_mean'] for fold in baseline_metrics]
    baseline_iou = [fold['overall']['iou_mean'] for fold in baseline_metrics]
    baseline_precision = [fold['overall']['precision_mean'] for fold in baseline_metrics]
    baseline_recall = [fold['overall']['recall_mean'] for fold in baseline_metrics]
    
    pp_dice = [fold['overall']['dice_mean'] for fold in postprocessed_metrics]
    pp_iou = [fold['overall']['iou_mean'] for fold in postprocessed_metrics]
    pp_precision = [fold['overall']['precision_mean'] for fold in postprocessed_metrics]
    pp_recall = [fold['overall']['recall_mean'] for fold in postprocessed_metrics]
    
    print("\n  COMPARISON (Postprocessed vs No Postprocessing):")
    print(f"    Dice improvement:      {np.mean(pp_dice) - np.mean(baseline_dice):+.4f}")
    print(f"    IoU improvement:       {np.mean(pp_iou) - np.mean(baseline_iou):+.4f}")
    print(f"    Precision improvement: {np.mean(pp_precision) - np.mean(baseline_precision):+.4f}")
    print(f"    Recall improvement:    {np.mean(pp_recall) - np.mean(baseline_recall):+.4f}")


def print_region_based_soma_comparison(
    baseline_metrics: List[Dict],
    postprocessed_metrics: List[Dict]
):
    """
    Print comparison of soma region-based metrics.
    
    Args:
        baseline_metrics: Metrics without postprocessing
        postprocessed_metrics: Metrics with postprocessing
    """
    if not (all('soma' in fold for fold in baseline_metrics) and 
            all('soma' in fold for fold in postprocessed_metrics)):
        return
    
    baseline_recall = [fold['soma']['recall_mean'] for fold in baseline_metrics]
    pp_recall = [fold['soma']['recall_mean'] for fold in postprocessed_metrics]
    
    print(f"  Recall improvement:   {np.mean(pp_recall) - np.mean(baseline_recall):+.4f}")


def print_region_based_branches_comparison(
    baseline_metrics: List[Dict],
    postprocessed_metrics: List[Dict]
):
    """
    Print comparison of branches region-based metrics.
    
    Args:
        baseline_metrics: Metrics without postprocessing
        postprocessed_metrics: Metrics with postprocessing
    """
    if not (all('branches' in fold for fold in baseline_metrics) and 
            all('branches' in fold for fold in postprocessed_metrics)):
        return
    
    baseline_dice = [fold['branches']['dice_mean'] for fold in baseline_metrics]
    baseline_iou = [fold['branches']['iou_mean'] for fold in baseline_metrics]
    baseline_precision = [fold['branches']['precision_mean'] for fold in baseline_metrics]
    baseline_recall = [fold['branches']['recall_mean'] for fold in baseline_metrics]
    
    pp_dice = [fold['branches']['dice_mean'] for fold in postprocessed_metrics]
    pp_iou = [fold['branches']['iou_mean'] for fold in postprocessed_metrics]
    pp_precision = [fold['branches']['precision_mean'] for fold in postprocessed_metrics]
    pp_recall = [fold['branches']['recall_mean'] for fold in postprocessed_metrics]
    
    print("\n  COMPARISON (Postprocessed vs No Postprocessing):")
    print(f"    Dice improvement:      {np.mean(pp_dice) - np.mean(baseline_dice):+.4f}")
    print(f"    IoU improvement:       {np.mean(pp_iou) - np.mean(baseline_iou):+.4f}")
    print(f"    Precision improvement: {np.mean(pp_precision) - np.mean(baseline_precision):+.4f}")
    print(f"    Recall improvement:    {np.mean(pp_recall) - np.mean(baseline_recall):+.4f}")


def print_region_based_metrics_with_comparison(
    baseline_metrics: List[Dict],
    postprocessed_metrics: List[Dict]
):
    """
    Print postprocessed region-based metrics with comparison to baseline.
    
    Args:
        baseline_metrics: Metrics without postprocessing
        postprocessed_metrics: Metrics with postprocessing
    """
    if not postprocessed_metrics:
        print("\nNo postprocessed region-based metrics were computed")
        return
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS (Region-Based Metrics - WITH POSTPROCESSING)")
    print("="*70)
    print(f"Based on {len(postprocessed_metrics)} folds with region annotations\n")
    
    # Overall metrics
    print_region_based_overall_metrics(postprocessed_metrics)
    if baseline_metrics:
        print_region_based_overall_comparison(baseline_metrics, postprocessed_metrics)
    
    # Soma metrics
    print_region_based_soma_metrics(postprocessed_metrics)
    if baseline_metrics:
        print_region_based_soma_comparison(baseline_metrics, postprocessed_metrics)
    
    # Branches metrics
    print_region_based_branches_metrics(postprocessed_metrics)
    if baseline_metrics:
        print_region_based_branches_comparison(baseline_metrics, postprocessed_metrics)
    
    print("="*70 + "\n")


def print_all_cross_validation_results(
    all_fold_metrics: List[Dict],
    all_fold_region_metrics: List[Dict],
    all_fold_postprocessed_metrics: Optional[List[Dict]] = None,
    all_fold_postprocessed_region_metrics: Optional[List[Dict]] = None,
    all_fold_postprocessed_probability_metrics: Optional[List[Dict]] = None,
    all_fold_postprocessed_probability_region_metrics: Optional[List[Dict]] = None
):
    """
    Print all cross-validation results in a comprehensive format.
    
    This is the main function that prints:
    1. Standard metrics without postprocessing
    2. Region-based metrics without postprocessing
    3. Standard metrics with postprocessing - ADAPTIVE (color/texture based)
    4. Region-based metrics with postprocessing - ADAPTIVE
    5. Standard metrics with postprocessing - PROBABILITY (UNet confidence based)
    6. Region-based metrics with postprocessing - PROBABILITY
    7. Comparisons between different postprocessing methods
    
    Args:
        all_fold_metrics: Standard metrics from all folds (no postprocessing)
        all_fold_region_metrics: Region-based metrics from all folds (no postprocessing)
        all_fold_postprocessed_metrics: Standard metrics with adaptive postprocessing (optional)
        all_fold_postprocessed_region_metrics: Region-based metrics with adaptive postprocessing (optional)
        all_fold_postprocessed_probability_metrics: Standard metrics with probability postprocessing (optional)
        all_fold_postprocessed_probability_region_metrics: Region-based metrics with probability postprocessing (optional)
    """
    # Print baseline results
    print_standard_metrics(all_fold_metrics, with_postprocessing=False)
    print_region_based_metrics(all_fold_region_metrics, with_postprocessing=False)
    
    # Print adaptive postprocessed results if available
    if all_fold_postprocessed_metrics:
        print("\n" + "="*70)
        print("POSTPROCESSING: ADAPTIVE (Color/Texture Based)")
        print("="*70)
        print_standard_metrics(all_fold_postprocessed_metrics, with_postprocessing=True, label="ADAPTIVE")
        print_standard_metrics_comparison(all_fold_metrics, all_fold_postprocessed_metrics)
    
    if all_fold_postprocessed_region_metrics:
        print_region_based_metrics_with_comparison(
            all_fold_region_metrics,
            all_fold_postprocessed_region_metrics
        )
    
    # Print probability-based postprocessed results if available
    if all_fold_postprocessed_probability_metrics:
        print("\n" + "="*70)
        print("POSTPROCESSING: PROBABILITY (UNet Confidence Based)")
        print("="*70)
        print_standard_metrics(all_fold_postprocessed_probability_metrics, with_postprocessing=True, label="PROBABILITY")
        print_standard_metrics_comparison(all_fold_metrics, all_fold_postprocessed_probability_metrics)
    
    if all_fold_postprocessed_probability_region_metrics:
        print_region_based_metrics_with_comparison(
            all_fold_region_metrics,
            all_fold_postprocessed_probability_region_metrics
        )
