import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _numeric_columns_present(df, columns):
    numeric_cols = []
    for c in columns:
        if c not in df.columns:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() >= 2:
            numeric_cols.append(c)
    return numeric_cols


def _categorical_columns_present(df, columns, numeric_cols):
    out = []
    numeric_set = set(numeric_cols)
    for c in columns:
        if c in df.columns and c not in numeric_set:
            out.append(c)
    return out


def _plot_heatmap(corr_df, title, figsize=(8, 6), cmap="coolwarm"):
    if corr_df.empty:
        print(f"{title}: no data to plot")
        return

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_df.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def scan_feature_correlation_matrix(df_summary_per_scan, scan_features, method="spearman"):
    numeric_scan_features = _numeric_columns_present(df_summary_per_scan, scan_features)
    if not numeric_scan_features:
        return pd.DataFrame()

    tmp = df_summary_per_scan[numeric_scan_features].apply(pd.to_numeric, errors="coerce")
    return tmp.corr(method=method)


def scan_vs_morphology_correlation_matrix(
    df_summary_per_scan,
    scan_features,
    morphology_features,
    method="spearman",
):
    numeric_scan_features = _numeric_columns_present(df_summary_per_scan, scan_features)
    morph_cols = [c for c in morphology_features if c in df_summary_per_scan.columns]
    if not numeric_scan_features or not morph_cols:
        return pd.DataFrame()

    left = df_summary_per_scan[numeric_scan_features].apply(pd.to_numeric, errors="coerce")
    right = df_summary_per_scan[morph_cols].apply(pd.to_numeric, errors="coerce")
    both = pd.concat([left, right], axis=1)
    corr = both.corr(method=method)
    return corr.loc[numeric_scan_features, morph_cols]


def plot_numeric_scan_features_by_diagnosis(
    df_summary_per_scan,
    scan_features,
    diagnosis_col="diagnosis_group",
    n_cols=3,
):
    numeric_scan_features = _numeric_columns_present(df_summary_per_scan, scan_features)
    if not numeric_scan_features:
        print("No numeric scan features found for boxplots.")
        return

    groups = [g for g in ["100+", "AD"] if g in df_summary_per_scan[diagnosis_col].dropna().unique()]
    if not groups:
        groups = sorted(df_summary_per_scan[diagnosis_col].dropna().astype(str).unique().tolist())

    n_rows = int(math.ceil(len(numeric_scan_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, feat in enumerate(numeric_scan_features):
        ax = axes[i]
        vals_by_group = []
        labels = []
        for g in groups:
            vals = pd.to_numeric(
                df_summary_per_scan.loc[df_summary_per_scan[diagnosis_col] == g, feat],
                errors="coerce",
            ).dropna()
            vals_by_group.append(vals)
            labels.append(g)

        ax.boxplot(vals_by_group, tick_labels=labels, showfliers=False)
        ax.set_title(feat)
        ax.set_ylabel("Value")

    for j in range(len(numeric_scan_features), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Numeric scan features by diagnosis group", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_categorical_scan_features_by_diagnosis(
    df_summary_per_scan,
    scan_features,
    diagnosis_col="diagnosis_group",
    normalize=True,
    n_cols=2,
):
    numeric_scan_features = _numeric_columns_present(df_summary_per_scan, scan_features)
    categorical_scan_features = _categorical_columns_present(df_summary_per_scan, scan_features, numeric_scan_features)

    if not categorical_scan_features:
        print("No categorical scan features found for bar charts.")
        return

    n_rows = int(math.ceil(len(categorical_scan_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, feat in enumerate(categorical_scan_features):
        ax = axes[i]
        ctab = pd.crosstab(df_summary_per_scan[diagnosis_col], df_summary_per_scan[feat], normalize="index" if normalize else False)
        ctab.plot(kind="bar", ax=ax)
        ax.set_title(feat)
        ax.set_xlabel(diagnosis_col)
        ax.set_ylabel("Proportion" if normalize else "Count")
        ax.legend(title=feat, bbox_to_anchor=(1.01, 1), loc="upper left")

    for j in range(len(categorical_scan_features), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Categorical scan features by diagnosis group", fontsize=14)
    plt.tight_layout()
    plt.show()


def run_basic_extra_variables_analysis(
    df_summary_per_scan,
    scan_features,
    morphology_features,
    diagnosis_col="diagnosis_group",
    corr_method="spearman",
):
    """
    Run a basic exploratory analysis of extra scan-level variables.

    Outputs
    -------
    dict with:
      - scan_feature_corr: correlation among numeric scan features
      - scan_vs_morph_corr: correlation between numeric scan features and morphology features
      - numeric_scan_features: detected numeric scan features
      - categorical_scan_features: detected categorical scan features
    """
    scan_feature_corr = scan_feature_correlation_matrix(
        df_summary_per_scan=df_summary_per_scan,
        scan_features=scan_features,
        method=corr_method,
    )
    print("=== Correlation: numeric scan features vs numeric scan features ===")
    print(scan_feature_corr.to_string())
    _plot_heatmap(scan_feature_corr, "Scan feature correlation matrix")

    scan_vs_morph_corr = scan_vs_morphology_correlation_matrix(
        df_summary_per_scan=df_summary_per_scan,
        scan_features=scan_features,
        morphology_features=morphology_features,
        method=corr_method,
    )
    print("=== Correlation: numeric scan features vs morphology features ===")
    print(scan_vs_morph_corr.to_string())
    _plot_heatmap(
        scan_vs_morph_corr,
        "Scan features vs morphology correlations",
        figsize=(max(8, len(scan_vs_morph_corr.columns) * 0.5), max(5, len(scan_vs_morph_corr.index) * 0.5)),
    )

    plot_numeric_scan_features_by_diagnosis(
        df_summary_per_scan=df_summary_per_scan,
        scan_features=scan_features,
        diagnosis_col=diagnosis_col,
    )

    plot_categorical_scan_features_by_diagnosis(
        df_summary_per_scan=df_summary_per_scan,
        scan_features=scan_features,
        diagnosis_col=diagnosis_col,
        normalize=True,
    )

    numeric_scan_features = _numeric_columns_present(df_summary_per_scan, scan_features)
    categorical_scan_features = _categorical_columns_present(df_summary_per_scan, scan_features, numeric_scan_features)

    return {
        "scan_feature_corr": scan_feature_corr,
        "scan_vs_morph_corr": scan_vs_morph_corr,
        "numeric_scan_features": numeric_scan_features,
        "categorical_scan_features": categorical_scan_features,
    }
