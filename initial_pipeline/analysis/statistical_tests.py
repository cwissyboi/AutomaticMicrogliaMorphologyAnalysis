import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist


DEFAULT_MORPHOLOGY_FEATURES = [
    "skeleton_length",
    "num_junctions",
    "num_components",
    "num_end_nodes",
    "num_start_nodes",
    "total_nodes",
    "end_to_start_ratio",
    "soma_area",
    "soma_perimeter",
    "soma_circularity",
    "cell_area",
    "cell_perimeter",
    "cell_convex_hull_area",
    "cell_convex_hull_perimeter",
    "cell_solidity",
    "cell_convexity",
    "cell_circularity",
    "cell_convex_circularity",
    "branch_area",
    "branch_perimeter",
    "sholl_min_radius",
    "sholl_peak_radius",
    "sholl_max_radius",
    "sholl_peak",
    "sholl_sum",
]


def filter_small_scans_by_cell_count(
    df_analysis,
    scan_col="scan_name",
    n_cells_col="n_cells",
    min_cells_per_scan=None,
    bottom_fraction=0.10,
):
    """
    Remove scans with too few cells.

    If min_cells_per_scan is provided, keep scans with n_cells >= threshold.
    Otherwise remove the bottom `bottom_fraction` of scans by cell count.
    """
    if scan_col not in df_analysis.columns:
        raise ValueError(f"Missing scan column: {scan_col}")

    # Prefer precomputed per-scan cell counts if present (e.g., in
    # final_cluster_props_df), otherwise fall back to counting rows.
    if n_cells_col in df_analysis.columns:
        scan_counts = (
            df_analysis[[scan_col, n_cells_col]]
            .dropna(subset=[scan_col, n_cells_col])
            .drop_duplicates(subset=[scan_col])
            .set_index(scan_col)[n_cells_col]
            .astype(float)
            .rename("n_cells")
            .sort_values()
        )
    else:
        scan_counts = (
            df_analysis.groupby(scan_col)
            .size()
            .rename("n_cells")
            .sort_values()
        )
    if scan_counts.empty:
        return {
            "df_filtered": df_analysis.copy(),
            "scan_counts": scan_counts,
            "threshold_used": np.nan,
            "n_scans_removed": 0,
        }

    if min_cells_per_scan is not None:
        threshold = int(min_cells_per_scan)
    else:
        if not (0 < bottom_fraction < 1):
            raise ValueError("bottom_fraction must be between 0 and 1")
        threshold = float(scan_counts.quantile(bottom_fraction))

    keep_scans = scan_counts[scan_counts >= threshold].index
    df_filtered = df_analysis[df_analysis[scan_col].isin(keep_scans)].copy()
    n_removed = int(scan_counts.index.difference(keep_scans).shape[0])

    return {
        "df_filtered": df_filtered,
        "scan_counts": scan_counts,
        "threshold_used": threshold,
        "n_scans_removed": n_removed,
    }


def select_representative_morphology_features_scan_level(
    scan_level_df,
    candidate_features,
    corr_method="spearman",
    corr_threshold=0.7,
):
    """
    Select one representative feature from each cluster of correlated features.

    Strategy
    --------
    1) Compute feature-feature correlation matrix at scan level.
    2) Cluster features with distance = 1 - abs(correlation).
    3) For each cluster, pick the most central feature (highest mean abs corr
       within that cluster).

    Parameters
    ----------
    scan_level_df : pd.DataFrame
        One row per scan.
    candidate_features : list[str]
        Morphology feature names to consider.
    corr_method : str
        Correlation method passed to pandas.DataFrame.corr.
    corr_threshold : float
        Features with abs(corr) roughly above this threshold will tend to be
        grouped together.

    Returns
    -------
    dict
        {
          'representative_features': list[str],
          'feature_cluster_map': pd.DataFrame,
          'correlation_matrix': pd.DataFrame,
        }
    """

    feats = [f for f in candidate_features if f in scan_level_df.columns]
    if not feats:
        raise ValueError("No candidate morphology features found in scan_level_df")

    X = scan_level_df[feats].copy()

    # Remove features that are unusable for correlation clustering.
    # - all NaN
    # - constant within scan-level table (no variance)
    usable = []
    dropped = []
    for f in feats:
        col = X[f]
        if col.isna().all():
            dropped.append((f, "all_nan"))
            continue
        if col.nunique(dropna=True) <= 1:
            dropped.append((f, "constant"))
            continue
        usable.append(f)

    if not usable:
        raise ValueError("All candidate morphology features were constant or NaN at scan level")

    if len(usable) == 1:
        feature_cluster_map = pd.DataFrame({"feature": usable, "feature_cluster": [1]})
        corr = pd.DataFrame([[1.0]], index=usable, columns=usable)
        return {
            "representative_features": usable,
            "feature_cluster_map": feature_cluster_map,
            "correlation_matrix": corr,
            "dropped_features": pd.DataFrame(dropped, columns=["feature", "drop_reason"]),
        }

    corr = X[usable].corr(method=corr_method).fillna(0.0)
    dist = 1.0 - corr.abs()
    np.fill_diagonal(dist.values, 0.0)

    Z = linkage(squareform(dist.values, checks=False), method="average")
    cluster_labels = fcluster(Z, t=(1.0 - corr_threshold), criterion="distance")

    feature_cluster_map = pd.DataFrame({
        "feature": usable,
        "feature_cluster": cluster_labels,
    }).sort_values(["feature_cluster", "feature"]).reset_index(drop=True)

    representatives = []
    for cid, g in feature_cluster_map.groupby("feature_cluster"):
        cluster_feats = g["feature"].tolist()
        if len(cluster_feats) == 1:
            representatives.append(cluster_feats[0])
            continue
        sub_corr = corr.loc[cluster_feats, cluster_feats].abs()
        centrality = sub_corr.mean(axis=1)
        representatives.append(centrality.idxmax())

    representatives = sorted(set(representatives))

    return {
        "representative_features": representatives,
        "feature_cluster_map": feature_cluster_map,
        "correlation_matrix": corr,
        "dropped_features": pd.DataFrame(dropped, columns=["feature", "drop_reason"]),
    }


def choose_best_morphology_features_for_testing(
    df_work,
    possible_morphology_features,
    scan_col,
    diagnosis_col,
    scan_features=None,
    corr_method="spearman",
    corr_threshold=0.7,
):
    """
    Select a representative morphology feature set from a constrained candidate list.

    The candidate pool is filtered *before* correlation clustering to include only
    features explicitly listed in possible_morphology_features and present in df_work.
    """
    if scan_features is None:
        scan_features = []

    # Filter first: only consider user-provided morphology subset.
    filtered_candidates = [
        f for f in possible_morphology_features
        if f in df_work.columns and pd.api.types.is_numeric_dtype(df_work[f])
    ]
    if not filtered_candidates:
        raise ValueError("No valid morphology features found from possible_morphology_features")

    # Build scan-level table for correlation-based representative selection.
    scan_level_tmp = aggregate_scan_level_morphology_df(
        df_work=df_work,
        morphology_features=filtered_candidates,
        diagnosis_col=diagnosis_col,
        scan_col=scan_col,
        scan_features=scan_features,
        warn_on_inconsistent=True,
    )

    selected = select_representative_morphology_features_scan_level(
        scan_level_df=scan_level_tmp,
        candidate_features=filtered_candidates,
        corr_method=corr_method,
        corr_threshold=corr_threshold,
    )
    return selected


def aggregate_scan_level_morphology_df(
    df_work,
    morphology_features,
    diagnosis_col="diagnosis_group",
    scan_col="scan_name",
    scan_features=None,
    warn_on_inconsistent=True,
):
    """
    Shared aggregation utility: one row per scan.

    - Morphology features are averaged within each scan.
    - diagnosis_col and scan_features are carried as first value per scan.
    """
    if scan_features is None:
        scan_features = []

    agg_map = {f: "mean" for f in morphology_features if f in df_work.columns}
    for c in [diagnosis_col] + list(scan_features):
        if c in df_work.columns:
            agg_map[c] = "first"

    scan_level_df = (
        df_work.groupby(scan_col, as_index=False)
        .agg(agg_map)
    )

    if warn_on_inconsistent:
        check_cols = [diagnosis_col] + list(scan_features)
        for c in check_cols:
            if c not in df_work.columns:
                continue
            nunique_per_scan = df_work.groupby(scan_col)[c].nunique(dropna=False)
            if (nunique_per_scan > 1).any():
                n_bad = int((nunique_per_scan > 1).sum())
                print(
                    f"Warning: column '{c}' has >1 distinct value in {n_bad} scans; "
                    "using first value during scan-level aggregation."
                )

    return scan_level_df


def mixed_effect_morphology_feature_tests(
    df_analysis,
    morphology_features=None,
    scan_features=None,
    diagnosis_col="diagnosis_group",
    scan_col="scan_name",
    group_a="AD",
    group_b="100+",
    max_cells_per_scan=None,
    random_state=42,
    apply_fdr=True,
    remove_small_scans=False,
    min_cells_per_scan_for_inclusion=None,
):
    """
    Build scan-level means and test morphology-feature differences between
    diagnosis groups while adjusting for scan-level covariates.

    Step 1 (aggregation):
      Collapse cell-level rows to one row per scan_name:
      - morphology features: mean within scan
      - diagnosis_group + scan_features: one value per scan (first value,
        after consistency checks)

    Step 2 (model per feature at scan level):
      y_s = beta_0 + beta_1 * I[group_s = AD] + sum_k gamma_k * X_{s,k} + epsilon_s
      where X_{s,k} are scan_features covariates (e.g., Sex).

    Why this helps:
      Aggregating to scan level avoids cell-level pseudo-replication and makes
      diagnosis inference happen at the scan/patient unit.

    Parameters
    ----------
    df_analysis : pd.DataFrame
        Must contain diagnosis_col, scan_col, and morphology feature columns.
    morphology_features : list[str] | None
        Feature columns to test. If None, infer numeric columns and exclude
        obvious metadata.
    scan_features : list[str] | None
        Scan-level covariates to include in the model (e.g., ['Sex']).
        These are carried into the scan-level dataframe without averaging.
    diagnosis_col : str
        Column with diagnosis labels.
    scan_col : str
        Column identifying scans (random-effect grouping variable).
    group_a, group_b : str
        Groups compared; reported effect is group_a - group_b.
    max_cells_per_scan : int | None
        Optional per-scan cap to keep fitting tractable on very large datasets.
        If None, use all rows.
    random_state : int
        RNG seed for optional subsampling.
    apply_fdr : bool
        If True, add Benjamini-Hochberg FDR q-values across features.
    remove_small_scans : bool
        If True, remove scans with low cell counts before aggregation/modeling.
    min_cells_per_scan_for_inclusion : int | None
        Used only when remove_small_scans=True.
        - If int: keep scans with >= this many cells.
        - If None: remove bottom 10% of scans by cell count.

    Returns
    -------
    dict
        {
          'scan_level_df': one row per scan with mean morphology + scan features,
          'model_results': one row per feature with adjusted diagnosis effect.
        }
    """

    # Import inside the function so other utilities in this module can still be
    # used even if statsmodels is unavailable in a given environment.
    import statsmodels.formula.api as smf

    if scan_features is None:
        scan_features = []

    required_cols = {diagnosis_col, scan_col, *scan_features}
    missing = required_cols - set(df_analysis.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep only the two requested diagnosis groups for a clean binary contrast.
    df_work = df_analysis[df_analysis[diagnosis_col].isin([group_a, group_b])].copy()
    if df_work.empty:
        raise ValueError(f"No rows found for groups {group_a} and {group_b}")

    if remove_small_scans:
        filtered = filter_small_scans_by_cell_count(
            df_work,
            scan_col=scan_col,
            min_cells_per_scan=min_cells_per_scan_for_inclusion,
            bottom_fraction=0.10,
        )
        df_work = filtered["df_filtered"]
        print(
            "Applied small-scan filter: "
            f"threshold={filtered['threshold_used']}, "
            f"removed_scans={filtered['n_scans_removed']}"
        )

    # max_cells_per_scan is kept for backward-compatible signature.
    # After scan-level aggregation it is not used.
    _ = max_cells_per_scan
    _ = random_state

    if morphology_features is None:
        selected = choose_best_morphology_features_for_testing(
            df_work=df_work,
            possible_morphology_features=DEFAULT_MORPHOLOGY_FEATURES,
            scan_col=scan_col,
            diagnosis_col=diagnosis_col,
            scan_features=scan_features,
            corr_method="spearman",
            corr_threshold=0.7,
        )
        morphology_features = selected["representative_features"]

        print("Selected representative morphology features (scan-level correlation clustering):")
        print(morphology_features)
        if not selected["dropped_features"].empty:
            print("Dropped features during selection (all_nan/constant):")
            print(selected["dropped_features"].to_string(index=False))

    # Build scan-level table (shared aggregation utility).
    scan_level_df = aggregate_scan_level_morphology_df(
        df_work=df_work,
        morphology_features=morphology_features,
        diagnosis_col=diagnosis_col,
        scan_col=scan_col,
        scan_features=scan_features,
        warn_on_inconsistent=True,
    )

    # Parameter name generated by statsmodels for the diagnosis fixed effect.
    # We set group_b as reference, so coefficient means (group_a - group_b).
    effect_name = f"C({diagnosis_col}, Treatment(reference='{group_b}'))[T.{group_a}]"

    rows = []
    for feature in morphology_features:
        if feature not in scan_level_df.columns:
            continue

        model_cols = [feature, diagnosis_col] + list(scan_features)
        sub = scan_level_df[model_cols].dropna().copy()
        if sub.empty:
            continue

        # Need both groups for diagnosis contrast.
        present_groups = set(sub[diagnosis_col].unique())
        if not ({group_a, group_b} <= present_groups):
            continue

        # Scan-level regression adjusted for scan_features.
        # Categorical covariates use C(...), numeric covariates stay numeric.
        cov_terms = []
        for c in scan_features:
            if c not in sub.columns:
                continue
            if pd.api.types.is_numeric_dtype(sub[c]):
                cov_terms.append(c)
            else:
                cov_terms.append(f"C({c})")

        rhs_terms = [f"C({diagnosis_col}, Treatment(reference='{group_b}'))"] + cov_terms
        formula = f"{feature} ~ {' + '.join(rhs_terms)}"

        try:
            model = smf.ols(formula=formula, data=sub)
            result = model.fit()

            coef = float(result.params.get(effect_name, np.nan))
            se = float(result.bse.get(effect_name, np.nan))
            z_stat = float(result.tvalues.get(effect_name, np.nan))
            p_val = float(result.pvalues.get(effect_name, np.nan))

            # Approximate 95% Wald CI for the fixed effect.
            ci_low = coef - 1.96 * se if pd.notna(se) else np.nan
            ci_high = coef + 1.96 * se if pd.notna(se) else np.nan

            rows.append(
                {
                    "feature": feature,
                    "n_scans": int(len(sub)),
                    f"coef_{group_a}_minus_{group_b}": coef,
                    "se": se,
                    "z_stat": z_stat,
                    "p_value": p_val,
                    "ci_2.5%": ci_low,
                    "ci_97.5%": ci_high,
                    "converged": bool(getattr(result, "converged", False)),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "feature": feature,
                    "n_scans": int(len(sub)),
                    f"coef_{group_a}_minus_{group_b}": np.nan,
                    "se": np.nan,
                    "z_stat": np.nan,
                    "p_value": np.nan,
                    "ci_2.5%": np.nan,
                    "ci_97.5%": np.nan,
                    "converged": False,
                    "fit_error": str(exc),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No features were successfully evaluated by mixed-effects modeling.")

    if apply_fdr:
        from statsmodels.stats.multitest import multipletests

        valid = out["p_value"].notna()
        out["q_value_fdr_bh"] = np.nan
        if valid.any():
            _, qvals, _, _ = multipletests(out.loc[valid, "p_value"].to_numpy(), method="fdr_bh")
            out.loc[valid, "q_value_fdr_bh"] = qvals

    sort_col = "q_value_fdr_bh" if apply_fdr and "q_value_fdr_bh" in out.columns else "p_value"
    out = out.sort_values(sort_col, na_position="last").reset_index(drop=True)

    print("=== Scan-level feature test (diagnosis adjusted for scan_features) ===")
    print(f"Scan-level dataframe shape: {scan_level_df.shape}")
    print(
        "Interpretation: coef > 0 means higher values in "
        f"{group_a} than {group_b} after adjusting for scan_features: {scan_features}."
    )
    print(out.to_string(index=False))

    return {
        "scan_level_df": scan_level_df,
        "model_results": out,
    }


def permanova_morphology_feature_tests(
    df_analysis,
    morphology_features=None,
    scan_features=None,
    diagnosis_col="diagnosis_group",
    scan_col="scan_name",
    group_a="AD",
    group_b="100+",
    random_state=42,
    remove_small_scans=False,
    min_cells_per_scan_for_inclusion=None,
    n_permutations=4999,
    standardize_features=True,
):
    """
    PERMANOVA on scan-level morphology profiles.

    Workflow
    --------
    1) Filter to group_a/group_b and optional small-scan removal.
    2) Aggregate to one row per scan (mean morphology; keep diagnosis + scan_features).
    3) Build a distance matrix from scan-level morphology vectors.
    4) Compute PERMANOVA pseudo-F and permutation p-value for diagnosis effect.

    Notes
    -----
    - This is a global multivariate test across all selected morphology features.
    - scan_features are preserved in scan_level_df for downstream checks but are
      not used as covariates in this one-factor PERMANOVA.
    """
    if scan_features is None:
        scan_features = []

    required_cols = {diagnosis_col, scan_col, *scan_features}
    missing = required_cols - set(df_analysis.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df_work = df_analysis[df_analysis[diagnosis_col].isin([group_a, group_b])].copy()
    if df_work.empty:
        raise ValueError(f"No rows found for groups {group_a} and {group_b}")

    if remove_small_scans:
        filtered = filter_small_scans_by_cell_count(
            df_work,
            scan_col=scan_col,
            min_cells_per_scan=min_cells_per_scan_for_inclusion,
            bottom_fraction=0.10,
        )
        df_work = filtered["df_filtered"]
        print(
            "Applied small-scan filter for PERMANOVA: "
            f"threshold={filtered['threshold_used']}, "
            f"removed_scans={filtered['n_scans_removed']}"
        )

    if morphology_features is None:
        selected = choose_best_morphology_features_for_testing(
            df_work=df_work,
            possible_morphology_features=DEFAULT_MORPHOLOGY_FEATURES,
            scan_col=scan_col,
            diagnosis_col=diagnosis_col,
            scan_features=scan_features,
            corr_method="spearman",
            corr_threshold=0.7,
        )
        morphology_features = selected["representative_features"]
        print("Selected representative morphology features for PERMANOVA:")
        print(morphology_features)

    scan_level_df = aggregate_scan_level_morphology_df(
        df_work=df_work,
        morphology_features=morphology_features,
        diagnosis_col=diagnosis_col,
        scan_col=scan_col,
        scan_features=scan_features,
        warn_on_inconsistent=True,
    )

    model_cols = [diagnosis_col] + morphology_features
    sub = scan_level_df[model_cols].dropna().copy()
    sub = sub[sub[diagnosis_col].isin([group_a, group_b])]
    if sub.empty:
        raise ValueError("No scan-level rows available for PERMANOVA after NA filtering")

    # Ensure both groups are represented and have at least 2 scans.
    g_counts = sub[diagnosis_col].value_counts()
    if group_a not in g_counts or group_b not in g_counts:
        raise ValueError("Both groups must be present in scan-level dataframe for PERMANOVA")
    if int(g_counts[group_a]) < 2 or int(g_counts[group_b]) < 2:
        raise ValueError("Need at least 2 scans per group for PERMANOVA")

    X = sub[morphology_features].to_numpy(dtype=float)
    if standardize_features:
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        std = np.where(std == 0, 1.0, std)
        X = (X - mean) / std

    groups = sub[diagnosis_col].astype(str).to_numpy()
    dist_matrix = squareform(pdist(X, metric="euclidean"))

    def _pseudo_f_and_r2(D, labels):
        # Distance-based sums of squares for one-factor PERMANOVA.
        n = D.shape[0]
        unique_groups = np.unique(labels)
        g = len(unique_groups)
        if g < 2:
            return np.nan, np.nan

        d2 = D ** 2
        ss_total = d2.sum() / (2.0 * n)

        ss_within = 0.0
        for grp in unique_groups:
            idx = np.where(labels == grp)[0]
            nk = len(idx)
            if nk <= 1:
                continue
            block = d2[np.ix_(idx, idx)]
            ss_within += block.sum() / (2.0 * nk)

        ss_between = ss_total - ss_within
        df_between = g - 1
        df_within = n - g
        if df_between <= 0 or df_within <= 0:
            return np.nan, np.nan

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        if ms_within <= 0:
            return np.nan, np.nan

        f_stat = ms_between / ms_within
        r2 = ss_between / ss_total if ss_total > 0 else np.nan
        return float(f_stat), float(r2)

    rng = np.random.default_rng(random_state)
    f_obs, r2_obs = _pseudo_f_and_r2(dist_matrix, groups)
    if pd.isna(f_obs):
        raise RuntimeError("Could not compute observed PERMANOVA pseudo-F")

    perm_f = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm_labels = rng.permutation(groups)
        f_i, _ = _pseudo_f_and_r2(dist_matrix, perm_labels)
        perm_f[i] = f_i if pd.notna(f_i) else -np.inf

    # +1 correction for permutation p-value
    p_val = (1.0 + np.sum(perm_f >= f_obs)) / (n_permutations + 1.0)

    result_df = pd.DataFrame(
        [
            {
                "n_scans": int(len(sub)),
                f"n_scans_{group_a}": int(g_counts[group_a]),
                f"n_scans_{group_b}": int(g_counts[group_b]),
                "n_features": int(len(morphology_features)),
                "pseudo_F": float(f_obs),
                "r2": float(r2_obs),
                "p_value_permutation": float(p_val),
                "n_permutations": int(n_permutations),
            }
        ]
    )

    print("=== PERMANOVA (scan-level morphology profiles) ===")
    print(result_df.to_string(index=False))

    return {
        "scan_level_df": scan_level_df,
        "permanova_result": result_df,
        "permutation_f_values": perm_f,
        "features_used": morphology_features,
    }

def dirichlet_scan_level_group_comparison(
    full_df,
    cluster_type="hard",                 # "hard" or "soft"
    diagnosis_col="diagnosis_group",
    scan_col="scan_name",
    group_a="AD",
    group_b="100+",
    n_boot=1000,
    random_state=42,
    eps=1e-10,
    remove_small_scans=False,
    min_cells_per_scan_for_inclusion=None,
):
    """
    Dirichlet analysis on scan-level compositions.

    full_df must contain:
      - diagnosis_col (e.g., diagnosis_group)
      - columns like f"{cluster_type}_cluster_0", ..., f"{cluster_type}_cluster_{K-1}"

    Optional robustness pre-filter:
      - If remove_small_scans=True, scans with low cell counts are removed
        before Dirichlet fitting.

    Returns dict with:
      - group_fits: fitted Dirichlet parameters and implied moments per group
      - mean_diff: bootstrap summary for mean composition differences (group_a - group_b)
      - var_diff: bootstrap summary for Dirichlet-implied variance differences
      - concentration_diff: bootstrap summary for alpha0 difference
    """

    # -------- helpers --------
    def _cluster_cols(df, prefix):
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            raise ValueError(f"No columns found with prefix '{prefix}'")
        cols = sorted(cols, key=lambda s: int(s.split("_")[-1]))
        return cols

    def _prepare_X(df_sub, cols):
        X = df_sub[cols].to_numpy(dtype=float)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("Need at least 2 cluster columns.")
        # clip and renormalize
        X = np.clip(X, eps, 1.0)
        X = X / X.sum(axis=1, keepdims=True)
        return X

    def _dirichlet_nll(log_alpha, X):
        alpha = np.exp(log_alpha)  # positivity
        a0 = alpha.sum()
        n = X.shape[0]
        ll = n * (gammaln(a0) - np.sum(gammaln(alpha))) + np.sum((alpha - 1.0) * np.log(X))
        return -ll

    def _dirichlet_init(X):
        # method-of-moments style init
        m = X.mean(axis=0)
        v = X.var(axis=0, ddof=1) if X.shape[0] > 1 else np.full_like(m, 1e-3)
        with np.errstate(divide="ignore", invalid="ignore"):
            a0_candidates = m * (1.0 - m) / np.maximum(v, 1e-12) - 1.0
        a0_candidates = a0_candidates[np.isfinite(a0_candidates) & (a0_candidates > 0)]
        a0 = np.median(a0_candidates) if len(a0_candidates) else 10.0
        alpha0 = np.maximum(m * a0, 1e-3)
        return alpha0

    def _fit_dirichlet(X):
        init_alpha = _dirichlet_init(X)
        res = minimize(
            _dirichlet_nll,
            x0=np.log(init_alpha),
            args=(X,),
            method="L-BFGS-B",
        )
        if not res.success:
            raise RuntimeError(f"Dirichlet fit failed: {res.message}")
        alpha = np.exp(res.x)
        a0 = alpha.sum()
        mu = alpha / a0
        var = (alpha * (a0 - alpha)) / (a0**2 * (a0 + 1.0))
        return alpha, a0, mu, var

    def _summarize_delta(samples, names, value_col):
        # samples: (B, K)
        rows = []
        for j, name in enumerate(names):
            d = samples[:, j]
            d = d[np.isfinite(d)]
            rows.append({
                "cluster": name,
                value_col: float(np.mean(d)),
                "ci_2.5%": float(np.quantile(d, 0.025)),
                "ci_97.5%": float(np.quantile(d, 0.975)),
                f"p_{group_a}_gt_{group_b}": float(np.mean(d > 0)),
            })
        out = pd.DataFrame(rows)
        out["abs_effect"] = out[value_col].abs()
        out = out.sort_values("abs_effect", ascending=False).drop(columns="abs_effect")
        return out

    # -------- data prep --------
    prefix = f"{cluster_type}_cluster_"
    cols = _cluster_cols(full_df, prefix)

    # Optional small-scan filtering for robustness checks.
    work_df = full_df.copy()
    if remove_small_scans:
        filtered = filter_small_scans_by_cell_count(
            work_df,
            scan_col=scan_col,
            min_cells_per_scan=min_cells_per_scan_for_inclusion,
            bottom_fraction=0.10,
        )
        work_df = filtered["df_filtered"]
        print(
            "Applied small-scan filter before Dirichlet: "
            f"threshold={filtered['threshold_used']}, "
            f"removed_scans={filtered['n_scans_removed']}"
        )

    df_use = work_df[[diagnosis_col] + cols].dropna().copy()
    df_a = df_use[df_use[diagnosis_col] == group_a]
    df_b = df_use[df_use[diagnosis_col] == group_b]

    if len(df_a) == 0 or len(df_b) == 0:
        raise ValueError(f"Need rows for both groups: {group_a}, {group_b}")

    X_a = _prepare_X(df_a, cols)
    X_b = _prepare_X(df_b, cols)

    # -------- fit on original data --------
    alpha_a, a0_a, mu_a, var_a = _fit_dirichlet(X_a)
    alpha_b, a0_b, mu_b, var_b = _fit_dirichlet(X_b)

    group_fits = pd.DataFrame({
        "group": [group_a, group_b],
        "n_scans": [len(X_a), len(X_b)],
        "alpha0_concentration": [a0_a, a0_b],
        "mean_entropy": [
            float(-np.sum(mu_a * np.log(mu_a + 1e-12))),
            float(-np.sum(mu_b * np.log(mu_b + 1e-12))),
        ],
    })

    # cluster-level fit summaries
    fit_detail = []
    for j, c in enumerate(cols):
        fit_detail.append({
            "cluster": c,
            f"mean_{group_a}": float(mu_a[j]),
            f"mean_{group_b}": float(mu_b[j]),
            "delta_mean_A_minus_B": float(mu_a[j] - mu_b[j]),
            f"dirichlet_var_{group_a}": float(var_a[j]),
            f"dirichlet_var_{group_b}": float(var_b[j]),
            "delta_var_A_minus_B": float(var_a[j] - var_b[j]),
        })
    fit_detail = pd.DataFrame(fit_detail)

    # -------- bootstrap uncertainty --------
    rng = np.random.default_rng(random_state)

    delta_mu_samples = []
    delta_var_samples = []
    delta_a0_samples = []

    for _ in range(n_boot):
        ia = rng.integers(0, len(X_a), len(X_a))
        ib = rng.integers(0, len(X_b), len(X_b))
        Xa_bs = X_a[ia]
        Xb_bs = X_b[ib]

        try:
            _, a0a_bs, mua_bs, vara_bs = _fit_dirichlet(Xa_bs)
            _, a0b_bs, mub_bs, varb_bs = _fit_dirichlet(Xb_bs)
        except Exception:
            continue

        delta_mu_samples.append(mua_bs - mub_bs)
        delta_var_samples.append(vara_bs - varb_bs)
        delta_a0_samples.append(a0a_bs - a0b_bs)

    if len(delta_mu_samples) < max(100, int(0.2 * n_boot)):
        raise RuntimeError(
            f"Too few successful bootstrap fits ({len(delta_mu_samples)}/{n_boot}). "
            "Try reducing n_boot or checking data quality."
        )

    delta_mu_samples = np.asarray(delta_mu_samples)      # (B_ok, K)
    delta_var_samples = np.asarray(delta_var_samples)    # (B_ok, K)
    delta_a0_samples = np.asarray(delta_a0_samples)      # (B_ok,)

    mean_diff = _summarize_delta(
        delta_mu_samples,
        names=cols,
        value_col=f"delta_mean_{group_a}_minus_{group_b}"
    )

    var_diff = _summarize_delta(
        delta_var_samples,
        names=cols,
        value_col=f"delta_dirichlet_var_{group_a}_minus_{group_b}"
    )

    concentration_diff = pd.DataFrame([{
        "metric": "alpha0_concentration",
        f"delta_{group_a}_minus_{group_b}": float(np.mean(delta_a0_samples)),
        "ci_2.5%": float(np.quantile(delta_a0_samples, 0.025)),
        "ci_97.5%": float(np.quantile(delta_a0_samples, 0.975)),
        f"p_{group_a}_gt_{group_b}": float(np.mean(delta_a0_samples > 0)),
    }])

    # -------- print concise summaries --------
    print(f"=== Dirichlet fit summary ({cluster_type}) ===")
    print(group_fits.to_string(index=False))
    print("\n=== Cluster mean differences (Dirichlet, bootstrap) ===")
    print(mean_diff.to_string(index=False))
    print("\n=== Cluster variance differences (Dirichlet-implied, bootstrap) ===")
    print(var_diff.to_string(index=False))
    print("\n=== Concentration difference (alpha0; higher = lower between-scan variance) ===")
    print(concentration_diff.to_string(index=False))

    return {
        "group_fits": group_fits,
        "fit_detail": fit_detail,
        "mean_diff": mean_diff,
        "var_diff": var_diff,
        "concentration_diff": concentration_diff,
        "bootstrap_successful": int(len(delta_mu_samples)),
        "cluster_cols": cols,
    }


def dirichlet_loocv_robustness_analysis(
    full_df,
    cluster_type="hard",
    diagnosis_col="diagnosis_group",
    scan_col="scan_name",
    group_a="AD",
    group_b="100+",
    n_boot=1000,
    random_state=42,
    eps=1e-10,
    only_check_ad=False,
    plot=True,
    verbose=False,
    significance_theshold=0.9,
):
    """
    Leave-one-out robustness analysis for Dirichlet scan-level group comparison.

    What this does
    --------------
    1) Runs the baseline Dirichlet analysis on all scans.
    2) Re-runs after dropping one scan at a time from group_a (e.g., AD).
    3) Re-runs after dropping one scan at a time from group_b (e.g., 100+),
       unless only_check_ad=True.
    4) Summarizes how key cluster-level estimates change across runs.

    Key outputs
    -----------
    - loocv_runs_df: long table with one row per (run, cluster)
    - loocv_summary_df: per-cluster robustness summary, including
      std/mean/min/max of delta and posterior probability
    - baseline_result: full baseline result from dirichlet_scan_level_group_comparison
    """
    import io
    import contextlib
    import matplotlib.pyplot as plt

    required_cols = {diagnosis_col, scan_col}
    missing = required_cols - set(full_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df_use = full_df[full_df[diagnosis_col].isin([group_a, group_b])].copy()
    if df_use.empty:
        raise ValueError(f"No rows found for groups {group_a} and {group_b}")

    scans_a = df_use.loc[df_use[diagnosis_col] == group_a, scan_col].dropna().unique().tolist()
    scans_b = df_use.loc[df_use[diagnosis_col] == group_b, scan_col].dropna().unique().tolist()
    if len(scans_a) < 2 or len(scans_b) < 2:
        raise ValueError("Need at least 2 scans per group for leave-one-out analysis")

    def _run_dirichlet(df_in, rs):
        if verbose:
            return dirichlet_scan_level_group_comparison(
                full_df=df_in,
                cluster_type=cluster_type,
                diagnosis_col=diagnosis_col,
                group_a=group_a,
                group_b=group_b,
                n_boot=n_boot,
                random_state=rs,
                eps=eps,
            )
        # Silence verbose prints from the underlying function during LOOCV.
        with contextlib.redirect_stdout(io.StringIO()):
            return dirichlet_scan_level_group_comparison(
                full_df=df_in,
                cluster_type=cluster_type,
                diagnosis_col=diagnosis_col,
                group_a=group_a,
                group_b=group_b,
                n_boot=n_boot,
                random_state=rs,
                eps=eps,
            )

    # Baseline fit (all scans)
    baseline_result = _run_dirichlet(df_use, random_state)
    baseline_mean = baseline_result["mean_diff"].copy()
    delta_col = f"delta_mean_{group_a}_minus_{group_b}"
    p_col = f"p_{group_a}_gt_{group_b}"
    if delta_col not in baseline_mean.columns or p_col not in baseline_mean.columns:
        raise RuntimeError("Expected Dirichlet output columns not found in baseline mean_diff")

    run_rows = []

    # Leave-one-out in group_a
    for i, scan_id in enumerate(scans_a):
        df_sub = df_use[~((df_use[diagnosis_col] == group_a) & (df_use[scan_col] == scan_id))].copy()
        res = _run_dirichlet(df_sub, random_state + 1000 + i)
        md = res["mean_diff"].copy()
        md["dropped_group"] = group_a
        md["dropped_scan"] = scan_id
        md["run_id"] = f"drop_{group_a}_{i}"
        run_rows.append(md)

    # Leave-one-out in group_b
    if not only_check_ad:
        for i, scan_id in enumerate(scans_b):
            df_sub = df_use[~((df_use[diagnosis_col] == group_b) & (df_use[scan_col] == scan_id))].copy()
            res = _run_dirichlet(df_sub, random_state + 2000 + i)
            md = res["mean_diff"].copy()
            md["dropped_group"] = group_b
            md["dropped_scan"] = scan_id
            md["run_id"] = f"drop_{group_b}_{i}"
            run_rows.append(md)

    loocv_runs_df = pd.concat(run_rows, ignore_index=True)

    # Robustness summaries by cluster and dropped group
    grp = loocv_runs_df.groupby(["cluster", "dropped_group"], as_index=False)
    threshold_label = str(significance_theshold).replace(".", "_")
    by_group_summary = grp.agg(
        n_runs=("run_id", "nunique"),
        delta_mean=(delta_col, "mean"),
        delta_std=(delta_col, "std"),
        delta_min=(delta_col, "min"),
        delta_max=(delta_col, "max"),
        p_mean=(p_col, "mean"),
        p_std=(p_col, "std"),
        p_min=(p_col, "min"),
        p_max=(p_col, "max"),
        **{
            f"frac_runs_p_gt_{threshold_label}": (
                p_col,
                lambda x: float(np.mean(x > significance_theshold)),
            )
        },
    )

    baseline_small = baseline_mean[["cluster", delta_col, p_col]].rename(
        columns={
            delta_col: "baseline_delta",
            p_col: "baseline_p",
        }
    )

    loocv_summary_df = by_group_summary.merge(baseline_small, on="cluster", how="left")
    loocv_summary_df["delta_shift_abs_mean"] = (loocv_summary_df["delta_mean"] - loocv_summary_df["baseline_delta"]).abs()
    loocv_summary_df["p_shift_abs_mean"] = (loocv_summary_df["p_mean"] - loocv_summary_df["baseline_p"]).abs()

    print(f"=== Dirichlet LOOCV robustness ({cluster_type}) ===")
    if only_check_ad:
        print(f"Scans: {group_a}={len(scans_a)} (LOOCV), {group_b}={len(scans_b)} (baseline only)")
    else:
        print(f"Scans: {group_a}={len(scans_a)}, {group_b}={len(scans_b)}")
    print("Baseline cluster mean differences:")
    print(baseline_mean.to_string(index=False))
    print("\nLOOCV summary (by dropped group):")
    print(loocv_summary_df.to_string(index=False))

    if plot:
        # Plot 1: delta stability (mean +/- std across LOOCV runs)
        clusters = baseline_small["cluster"].tolist()
        x = np.arange(len(clusters), dtype=float)
        offset = 0.12

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        plot_groups = [(group_a, -1, "tab:red")]
        if not only_check_ad:
            plot_groups.append((group_b, 1, "tab:blue"))

        for g, sign, color in plot_groups:
            d = loocv_summary_df[loocv_summary_df["dropped_group"] == g].set_index("cluster").reindex(clusters)
            axes[0].errorbar(
                x + sign * offset,
                d["delta_mean"],
                yerr=d["delta_std"],
                fmt="o",
                capsize=3,
                color=color,
                label=f"Drop-1 from {g}",
            )

        axes[0].plot(x, baseline_small["baseline_delta"], "ks-", label="Baseline")
        axes[0].axhline(0, color="gray", linewidth=1)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(clusters, rotation=45, ha="right")
        axes[0].set_title("LOOCV stability: delta mean")
        axes[0].set_ylabel(f"{delta_col}")
        axes[0].legend()

        # Plot 2: posterior directional probability stability
        for g, sign, color in plot_groups:
            d = loocv_summary_df[loocv_summary_df["dropped_group"] == g].set_index("cluster").reindex(clusters)
            axes[1].errorbar(
                x + sign * offset,
                d["p_mean"],
                yerr=d["p_std"],
                fmt="o",
                capsize=3,
                color=color,
                label=f"Drop-1 from {g}",
            )

        axes[1].plot(x, baseline_small["baseline_p"], "ks-", label="Baseline")
        axes[1].axhline(0.5, color="gray", linewidth=1, linestyle="--")
        axes[1].set_ylim(0, 1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(clusters, rotation=45, ha="right")
        axes[1].set_title("LOOCV stability: directional probability")
        axes[1].set_ylabel(p_col)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return {
        "baseline_result": baseline_result,
        "loocv_runs_df": loocv_runs_df,
        "loocv_summary_df": loocv_summary_df,
    }
