"""task1_eval -- evaluation pipeline for Task 1 (melody generation).

Modules:
    melody       -- the internal melody format + a builder.
    features     -- turn a melody into feature numbers.
    distribution -- compare two sets of melodies, feature by feature.
    copying      -- check whether the model is memorizing training data.
    report       -- tables and plots that summarize an evaluation run.
    pipeline     -- run the whole evaluation in one call (evaluate()).

Quick start:
    import task1_eval
    from task1_eval import melody

    # build melodies from (pitch, duration) pairs
    generated  = [melody.melody_from_durations(pairs) for pairs in ...]
    real_test  = [melody.melody_from_durations(pairs) for pairs in ...]
    real_train = [melody.melody_from_durations(pairs) for pairs in ...]

    result = task1_eval.evaluate(generated, real_test, real_train)
    result.distances        # the main table
"""

from .melody import Note, melody_from_durations
from . import features
from .features import extract_all, extract_set
from . import distribution
from .distribution import (
    feature_distance,
    feature_scales,
    compare_sets,
    mean_distance,
)
from . import copying
from .copying import ngram_overlap, overlap_per_melody, overlap_curve
from . import report
from .report import (
    comparison_table,
    plot_feature_hists,
    plot_all_features,
    copying_table,
)
from . import pipeline
from .pipeline import evaluate, EvalResult

__all__ = [
    "Note",
    "melody_from_durations",
    "features",
    "extract_all",
    "extract_set",
    "distribution",
    "feature_distance",
    "feature_scales",
    "compare_sets",
    "mean_distance",
    "copying",
    "ngram_overlap",
    "overlap_per_melody",
    "overlap_curve",
    "report",
    "comparison_table",
    "plot_feature_hists",
    "plot_all_features",
    "copying_table",
    "pipeline",
    "evaluate",
    "EvalResult",
]
