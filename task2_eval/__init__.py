"""task2_eval -- evaluation pipeline for Task 2 (chord generation / harmonization).

Modules:
    chord_io     -- read melodies and chord sequences.
    data_split   -- the train/val/test split and POP909 file layout.
    chords       -- chord label utilities (label -> pitch classes etc.).
    agreement    -- score predicted chords against real chords (mir_eval).
    harmonicity  -- chord-and-melody fit ('good but different' metric).
    progression  -- chord-bigram plausibility of a predicted sequence.
    baselines    -- most-common / rule-based / melody-ignoring baselines.
    report       -- tables and plots summarizing the evaluation.
    pipeline     -- run the whole evaluation in one call (evaluate()).

Quick start:
    import task2_eval
    test_ids = task2_eval.read_ids("test_ids.txt")
    test_songs = task2_eval.load_songs_by_id("data", test_ids)
    train_ids = task2_eval.read_ids("train_ids.txt")
    train_songs = task2_eval.load_songs_by_id("data", train_ids)
    train_chords = [ch for (_, ch) in train_songs.values()]

    result = task2_eval.evaluate(test_songs, train_chords,
                                 model_predictions=...,
                                 hmm_predictions=...)
    result.table                 # the main side-by-side table
"""

from .chord_io import (
    Note,
    Chord,
    load_melody,
    load_chords,
    chords_from_grid,
)
from .data_split import (
    read_ids,
    load_split,
    pop909_midi_path,
    pop909_chord_path,
    filter_available,
    load_song,
    load_songs_by_id,
)
from . import chords
from .chords import (
    is_no_chord,
    pitch_classes,
    root_pitch_class,
    chord_vocabulary,
)
from . import agreement
from .agreement import score_song, score_set, SCORE_NAMES
from . import harmonicity
from .harmonicity import harmonicity_score, harmonicity_set
from . import progression
from .progression import fit_bigram, sequence_log_prob, plausibility_set
from . import baselines
from .baselines import (
    most_common_baseline,
    rule_based_baseline,
    melody_ignoring_baseline,
    build_baselines,
)
from . import report
from .report import comparison_table, plot_metric_bars, per_song_table
from . import pipeline
from .pipeline import evaluate, EvalResult

__all__ = [
    "Note",
    "Chord",
    "load_melody",
    "load_chords",
    "chords_from_grid",
    "read_ids",
    "load_split",
    "pop909_midi_path",
    "pop909_chord_path",
    "filter_available",
    "load_song",
    "load_songs_by_id",
    "chords",
    "is_no_chord",
    "pitch_classes",
    "root_pitch_class",
    "chord_vocabulary",
    "agreement",
    "score_song",
    "score_set",
    "SCORE_NAMES",
    "harmonicity",
    "harmonicity_score",
    "harmonicity_set",
    "progression",
    "fit_bigram",
    "sequence_log_prob",
    "plausibility_set",
    "baselines",
    "most_common_baseline",
    "rule_based_baseline",
    "melody_ignoring_baseline",
    "build_baselines",
    "report",
    "comparison_table",
    "plot_metric_bars",
    "per_song_table",
    "pipeline",
    "evaluate",
    "EvalResult",
]
