"""task1_eval.pipeline -- run the whole Task 1 evaluation in one call.

This ties together features, distribution and copying. The individual
modules stay usable on their own; this is just the convenient top-level
entry point for the notebook.
"""

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import extract_set
from .distribution import feature_scales, compare_sets
from .copying import overlap_curve
from .report import comparison_table, copying_table


@dataclass
class EvalResult:
    """The result of one evaluation run.

    Attributes:
        counts    -- dict: how many melodies were in each set.
        distances -- DataFrame: per-feature distances (floor / model /
                     baseline), with a MEAN row. This is the main result.
        copying   -- DataFrame: n-gram overlap with the training set.
        features  -- dict {set name -> extract_set result}; pass to
                     report.plot_all_features to draw the histograms.
        scale     -- dict: the per-feature normalization used.
    """
    counts: dict
    distances: Any
    copying: Any
    features: dict
    scale: dict

    def __repr__(self):
        return f"EvalResult(counts={self.counts})"


def _floor_distance(real_train, f_test, scale, sample_size, repeats, seed):
    """The real-vs-real baseline distance.

    Instead of using ALL training melodies, this draws random subsets of
    `sample_size` melodies and measures each subset's distance to the
    real test set. The subset is meant to be the same size as the
    generated set, so the baseline carries the same small-sample noise
    as the model comparison -- a fair "floor".

    The result is averaged over `repeats` random draws for stability.

    Returns: dict feature name -> averaged distance.
    """
    rng = random.Random(seed)
    n = min(sample_size, len(real_train))
    per_draw = []
    for _ in range(repeats):
        subset = rng.sample(real_train, n)
        per_draw.append(compare_sets(extract_set(subset), f_test, scale=scale))
    feature_names = per_draw[0].keys()
    return {
        name: float(np.nanmean([draw[name] for draw in per_draw]))
        for name in feature_names
    }


def evaluate(generated, real_test, real_train, baseline=None,
             copy_ns=(3, 4, 5, 6, 8),
             floor_sample_size=150, floor_repeats=5, seed=0):
    """Run the full Task 1 evaluation.

    Melody-list inputs (build them with melody.melody_from_durations):
        generated  -- melodies from the model being evaluated
        real_test  -- real POP909 melodies; the comparison target
        real_train -- real POP909 melodies; the pool the floor is drawn from
        baseline   -- (optional) melodies from a trivial baseline model
                      (e.g. a unigram pitch sampler), shown as its own
                      column to demonstrate the model beats it.

    Floor settings:
        floor_sample_size -- how many training melodies to draw for the
                             real-vs-real baseline. For the fairest
                             comparison, set this close to len(generated).
        floor_repeats     -- number of random draws to average over.
        seed              -- makes the random draws reproducible.

    Returns an EvalResult. See its docstring for the fields.
    """
    # 1. Feature numbers for each set.
    f_gen = extract_set(generated)
    f_test = extract_set(real_test)
    f_train = extract_set(real_train)  # full set, used only for the plots

    # 2. Distances, all normalized by the real test set's spread so the
    #    floor / model / baseline columns are directly comparable.
    scale = feature_scales(f_test)
    comparisons = {
        # real vs real: random training subsets, averaged (the baseline floor)
        "floor": _floor_distance(
            real_train, f_test, scale, floor_sample_size, floor_repeats, seed
        ),
        # generated vs real (the check)
        "model": compare_sets(f_gen, f_test, scale=scale),
    }
    features_by_set = {
        "real test": f_test,
        "real train": f_train,
        "model": f_gen,
    }

    # 3. Copying check: overlap with the (full) training set. real-test-
    #    vs-train is the natural reference -- real songs share short
    #    phrases too, and both rows use the same full training pool.
    curves = {
        "model vs train": overlap_curve(generated, real_train, ns=copy_ns),
        "real test vs train": overlap_curve(real_test, real_train, ns=copy_ns),
    }

    counts = {
        "generated": len(generated),
        "real_test": len(real_test),
        "real_train": len(real_train),
        "floor_sample_size": min(floor_sample_size, len(real_train)),
    }

    # 4. Optionally fold in a trivial baseline model.
    if baseline is not None:
        f_base = extract_set(baseline)
        comparisons["baseline"] = compare_sets(f_base, f_test, scale=scale)
        features_by_set["baseline"] = f_base
        curves["baseline vs train"] = overlap_curve(baseline, real_train,
                                                    ns=copy_ns)
        counts["baseline"] = len(baseline)

    return EvalResult(
        counts=counts,
        distances=comparison_table(comparisons),
        copying=copying_table(curves),
        features=features_by_set,
        scale=scale,
    )
