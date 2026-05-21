"""task2_eval.pipeline -- run the whole Task 2 evaluation in one call.

This ties agreement + harmonicity + progression + baselines together.
The individual modules stay usable on their own; this is just the
convenient top-level entry point for the notebook.
"""

from dataclasses import dataclass
from typing import Any

from .agreement import score_set
from .harmonicity import harmonicity_set
from .progression import fit_bigram, plausibility_set
from .baselines import build_baselines
from .report import comparison_table


@dataclass
class EvalResult:
    """The result of one Task 2 evaluation run.

    Attributes:
        counts      -- dict: songs in each set, methods evaluated.
        table       -- DataFrame: the main side-by-side comparison
                       (rows = methods, columns = metrics).
        agreement   -- dict {method -> {score_name -> value}}
        harmonicity -- dict {method -> value}
        progression -- dict {method -> value}
        per_song    -- dict {metric -> {method -> {song_id -> value}}}
                       for digging into per-song detail.
    """
    counts: dict
    table: Any
    agreement: dict
    harmonicity: dict
    progression: dict
    per_song: dict

    def __repr__(self):
        return f"EvalResult(counts={self.counts})"


def evaluate(test_songs, training_chord_seqs,
             model_predictions=None, hmm_predictions=None,
             include_baselines=True, seed=0):
    """Run the full Task 2 evaluation.

    Inputs:
        test_songs          -- {song_id -> (melody, real_chords)}
                               (use task2_eval.load_songs_by_id)
        training_chord_seqs -- iterable of real training ChordSeqs;
                               used to fit the bigram model AND the
                               most-common / melody-ignoring baselines.
        model_predictions   -- {song_id -> ChordSeq} from the main model
                               (optional; you can run with baselines only).
        hmm_predictions     -- {song_id -> ChordSeq} from the HMM baseline
                               (optional).
        include_baselines   -- True: also run the three simple baselines
                               (most_common / rule_based / melody_ignoring).
        seed                -- random seed for melody_ignoring sampling.

    Returns: an EvalResult.
    """
    # We iterate training_chord_seqs multiple times -- materialise to a
    # list so a generator doesn't get consumed on the first pass.
    training_chord_seqs = list(training_chord_seqs)

    # The real chord sequence for each test song is the reference for
    # both agreement (comparison target) and harmonicity (the melody
    # comes from the song).
    real_dict = {sid: ch for sid, (_, ch) in test_songs.items()}

    # Assemble every method to evaluate.
    methods = {"real": real_dict}
    if model_predictions is not None:
        methods["model"] = model_predictions
    if hmm_predictions is not None:
        methods["hmm"] = hmm_predictions
    if include_baselines:
        methods.update(build_baselines(test_songs, training_chord_seqs,
                                       seed=seed))

    # Bigram model used for the progression metric. Fit once, reuse for
    # every method below.
    bigram_model = fit_bigram(training_chord_seqs)

    # Score every method on every metric.
    agreement_agg, agreement_ps = {}, {}
    harmonicity_agg, harmonicity_ps = {}, {}
    progression_agg, progression_ps = {}, {}

    for name, preds in methods.items():
        # 1) agreement (predicted vs real)
        agr = score_set(real_dict, preds)
        agreement_agg[name] = agr["aggregate"]
        agreement_ps[name] = agr["per_song"]

        # 2) harmonicity (chords vs the real melody)
        pairs = {sid: (mel, preds[sid])
                 for sid, (mel, _) in test_songs.items()
                 if sid in preds}
        hmn = harmonicity_set(pairs)
        harmonicity_agg[name] = hmn["aggregate"]
        harmonicity_ps[name] = hmn["per_song"]

        # 3) progression (sequence plausibility under the trained bigram)
        prog = plausibility_set(preds, bigram_model)
        progression_agg[name] = prog["aggregate"]
        progression_ps[name] = prog["per_song"]

    table = comparison_table(
        agreement_by_method=agreement_agg,
        harmonicity_by_method=harmonicity_agg,
        progression_by_method=progression_agg,
    )

    counts = {
        "test_songs": len(test_songs),
        "training_chord_seqs": len(training_chord_seqs),
        "methods": list(methods.keys()),
    }
    per_song = {
        "agreement": agreement_ps,
        "harmonicity": harmonicity_ps,
        "progression": progression_ps,
    }

    return EvalResult(
        counts=counts,
        table=table,
        agreement=agreement_agg,
        harmonicity=harmonicity_agg,
        progression=progression_agg,
        per_song=per_song,
    )
