"""task1_eval.distribution -- compare two sets of melodies, feature by feature.

The idea (see the Task 1 eval plan): two sets of melodies are compared by
measuring, for each feature, how far apart the two sets' value
distributions are. We use the Wasserstein ("earth mover's") distance --
one clear number per feature, with no histogram-binning choices to make.

Typical use:
    real_test  = features.extract_set(real_test_melodies)
    real_train = features.extract_set(real_train_melodies)
    model      = features.extract_set(model_melodies)

    scale   = feature_scales(real_test)        # makes features comparable
    d_floor = compare_sets(real_train, real_test, scale=scale)
    d_model = compare_sets(model,      real_test, scale=scale)
"""

import numpy as np
from scipy.stats import wasserstein_distance


def _clean(values):
    """Drop nan/inf from a list of feature values, return a float array."""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def feature_distance(values_a, values_b):
    """Wasserstein distance between two lists of values for one feature.

    nan/inf values are dropped first. Returns nan if either side has no
    valid values left.
    """
    a = _clean(values_a)
    b = _clean(values_b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def feature_scales(features):
    """A per-feature scale (standard deviation) used for normalization.

    Pass the result as `scale` to compare_sets so distances are measured
    in units of the real data's natural spread -- this makes different
    features comparable to each other.

    Returns: dict feature name -> std. A zero std is replaced by 1.0 to
    avoid division by zero.
    """
    scales = {}
    for name, values in features.items():
        std = float(np.std(_clean(values)))
        scales[name] = std if std > 0 else 1.0
    return scales


def compare_sets(features_a, features_b, scale=None):
    """Compare two feature sets (each the output of features.extract_set).

    For every feature present in both sets, compute the Wasserstein
    distance. If `scale` is given (a dict of feature -> positive number),
    each distance is divided by that feature's scale.

    Returns: dict feature name -> distance.
    """
    shared = [name for name in features_a if name in features_b]
    result = {}
    for name in shared:
        dist = feature_distance(features_a[name], features_b[name])
        if scale is not None and name in scale:
            dist = dist / scale[name]
        result[name] = dist
    return result


def mean_distance(comparison):
    """Average distance across features -- a single summary number.

    `comparison` is the dict returned by compare_sets. nan distances are
    ignored.
    """
    vals = _clean(list(comparison.values()))
    return float(np.mean(vals)) if vals.size else float("nan")
