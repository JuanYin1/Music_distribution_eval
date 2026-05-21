"""task2_eval.report -- tables and plots summarizing a Task 2 run.

Functions return pandas DataFrames and matplotlib Axes; the notebook
decides how to display them. Nothing is printed or shown automatically,
so these are safe to import and reuse.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .agreement import SCORE_NAMES


def comparison_table(agreement_by_method=None,
                     harmonicity_by_method=None,
                     progression_by_method=None,
                     methods=None):
    """Build the main side-by-side comparison table.

    Each input is a dict keyed by method name (e.g. 'real', 'model',
    'hmm', 'most_common', 'rule_based', 'melody_ignoring'):

        agreement_by_method   -- {method -> {score_name -> float}}
                                 (the 'aggregate' from agreement.score_set)
        harmonicity_by_method -- {method -> float}
                                 (the 'aggregate' from harmonicity.harmonicity_set)
        progression_by_method -- {method -> float}
                                 (the 'aggregate' from progression.plausibility_set)

    Any input can be None or partially filled. `methods` optionally
    fixes the row order; otherwise rows are sorted alphabetically with
    'real' placed first if it appears.

    Returns a DataFrame: rows = methods, columns = each agreement
    strictness level + 'harmonicity' + 'progression'.
    """
    agreement_by_method = agreement_by_method or {}
    harmonicity_by_method = harmonicity_by_method or {}
    progression_by_method = progression_by_method or {}

    if methods is None:
        all_methods = (set(agreement_by_method)
                       | set(harmonicity_by_method)
                       | set(progression_by_method))
        methods = sorted(all_methods)
        if "real" in methods:
            methods = ["real"] + [m for m in methods if m != "real"]

    rows = {}
    for method in methods:
        row = {}
        for score in SCORE_NAMES:
            row[score] = agreement_by_method.get(method, {}).get(score, np.nan)
        row["harmonicity"] = harmonicity_by_method.get(method, np.nan)
        row["progression"] = progression_by_method.get(method, np.nan)
        rows[method] = row
    return pd.DataFrame(rows).T


def plot_metric_bars(table, metrics=None, ax=None):
    """Bar plot of selected metric columns across the methods in `table`.

    `table` is the output of comparison_table. `metrics` defaults to
    ['triads', 'harmonicity', 'progression'] -- the three headline
    numbers, one per pillar of the eval plan.

    Returns the matplotlib Axes.
    """
    if metrics is None:
        metrics = ["triads", "harmonicity", "progression"]
    sub = table[metrics]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sub.plot(kind="bar", ax=ax)
    ax.set_ylabel("score")
    ax.set_title("metric scores by method")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelrotation=30)
    return ax


def per_song_table(per_song_by_method):
    """Per-song values for ONE metric across several methods.

    `per_song_by_method` is {method_name -> {song_id -> value}}. The
    returned DataFrame is indexed by song_id with one column per method,
    useful for spotting songs where the model does much better or worse
    than the baselines.
    """
    return pd.DataFrame(per_song_by_method).rename_axis(index="song_id")
