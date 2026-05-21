"""task1_eval.report -- tables and plots that summarize an evaluation run.

Functions here return pandas DataFrames and matplotlib Figures/Axes; the
notebook decides how to display them. Nothing is printed or shown
automatically, so these are safe to import and reuse.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .features import SCALAR_FEATURES


def comparison_table(comparisons):
    """Build a feature-by-feature distance table.

    `comparisons` maps a column name to a distance dict (the output of
    distribution.compare_sets), e.g.:

        comparison_table({
            "floor": d_floor,   # real      vs real
            "model": d_model,   # generated vs real
            "hmm":   d_hmm,     # baseline  vs real
        })

    Returns a DataFrame: one row per feature, one column per comparison,
    plus a final 'MEAN' row.
    """
    df = pd.DataFrame(comparisons)
    df.loc["MEAN"] = df.mean(skipna=True)
    return df


def plot_feature_hists(features_by_set, feature_name, bins=20, ax=None):
    """Overlay the histogram of one feature across several sets.

    `features_by_set` maps a set name to an extract_set() result, e.g.:
        {"real test": f_test, "model": f_gen, "hmm": f_hmm}

    Returns the matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    for name, feats in features_by_set.items():
        values = np.asarray(feats[feature_name], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(values, bins=bins, density=True, alpha=0.5, label=name)
    ax.set_title(feature_name)
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    return ax


def plot_all_features(features_by_set, bins=20, ncols=3):
    """A grid of histograms, one subplot per scalar feature.

    Returns the matplotlib Figure.
    """
    names = list(SCALAR_FEATURES)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, feature_name in zip(axes, names):
        plot_feature_hists(features_by_set, feature_name, bins=bins, ax=ax)
    for ax in axes[len(names):]:        # hide any unused subplots
        ax.axis("off")
    fig.tight_layout()
    return fig


def copying_table(curves):
    """Build an n-gram-overlap table.

    `curves` maps a row name to an overlap curve (the dict returned by
    copying.overlap_curve), e.g.:

        copying_table({
            "model vs train":     curve_model,
            "real test vs train": curve_real,
        })

    Returns a DataFrame: one row per set, one column per n.
    """
    return pd.DataFrame(curves).T.sort_index(axis=1)
