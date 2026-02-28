"""
Utilities for printing cross-validation results in a formatted way.

This module provides functions to print standard metrics, region-based metrics,
and comparison between postprocessed and non-postprocessed results.
"""

import numpy as np
from typing import List, Dict, Optional


def print_standard_metrics(
    all_fold_metrics: List[Dict],
    with_postprocessing: bool = False
):
    """
    Print standard metrics (Dice, IoU, Morphology) across all folds.
    
    Args:
        all_fold_metrics: List of dictionaries containing metrics from each fold
        with_postprocessing: Whether these results are from postprocessed predictions
    """
    dice_scores = [m["dice"] for m in all_fold_metrics]
    iou_scores = [m["iou"] for m in all_fold_metrics]
    morphology_scores = [m["morphology"] for m in all_fold_metrics]
    
    label = "WITH POSTPROCESSING" if with_postprocessing else "NO POSTPROCESSING"
    print("\n" + "="*70)
    print(f"CROSS-VALIDATION RESULTS (Standard Metrics - {label})")
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
    all_fold_postprocessed_region_metrics: Optional[List[Dict]] = None
):
    """
    Print all cross-validation results in a comprehensive format.
    
    This is the main function that prints:
    1. Standard metrics without postprocessing
    2. Region-based metrics without postprocessing
    3. Standard metrics with postprocessing (if available)
    4. Region-based metrics with postprocessing (if available)
    5. Comparisons between postprocessed and non-postprocessed results
    
    Args:
        all_fold_metrics: Standard metrics from all folds (no postprocessing)
        all_fold_region_metrics: Region-based metrics from all folds (no postprocessing)
        all_fold_postprocessed_metrics: Standard metrics with postprocessing (optional)
        all_fold_postprocessed_region_metrics: Region-based metrics with postprocessing (optional)
    """
    # Print baseline results
    print_standard_metrics(all_fold_metrics, with_postprocessing=False)
    print_region_based_metrics(all_fold_region_metrics, with_postprocessing=False)
    
    # Print postprocessed results if available
    if all_fold_postprocessed_metrics:
        print_standard_metrics(all_fold_postprocessed_metrics, with_postprocessing=True)
        print_standard_metrics_comparison(all_fold_metrics, all_fold_postprocessed_metrics)
    
    if all_fold_postprocessed_region_metrics:
        print_region_based_metrics_with_comparison(
            all_fold_region_metrics,
            all_fold_postprocessed_region_metrics
        )
