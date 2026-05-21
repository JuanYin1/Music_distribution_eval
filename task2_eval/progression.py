"""task2_eval.progression -- how natural is a chord sequence?

We train a simple chord-bigram model on real training chord sequences,
then score predicted sequences under it: for each consecutive pair, look
up how likely 'previous chord -> next chord' was in the real data.
Higher average log-probability = transitions more like real POP909.

This catches the case where each chord might fit the melody just fine,
but the *order* of the chords doesn't make musical sense (e.g. random
jumps between unrelated chords). It's the third pillar of the eval
plan, alongside agreement.py (matches the reference?) and
harmonicity.py (fits the melody?).
"""

import math
from collections import defaultdict

import numpy as np


def fit_bigram(chord_seqs, smoothing=1.0):
    """Fit an add-k smoothed chord bigram model on real chord sequences.

    `chord_seqs` is an iterable of list[Chord]. `smoothing` is the
    Laplace k (default 1.0); unseen 'prev->curr' pairs get a small
    probability instead of zero.

    Returns a dict that sequence_log_prob() understands.
    """
    transitions = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    vocab = set()
    for seq in chord_seqs:
        labels = [c.label for c in seq]
        vocab.update(labels)
        for a, b in zip(labels, labels[1:]):
            transitions[a][b] += 1
            totals[a] += 1
    return {
        "transitions": {a: dict(d) for a, d in transitions.items()},
        "totals": dict(totals),
        "vocab": sorted(vocab),
        "smoothing": float(smoothing),
    }


def _log_prob_pair(prev, curr, model):
    """Smoothed log P(curr | prev) under the bigram model.

    For a previous label that was never seen in training, we fall back
    to a uniform distribution over the training vocabulary.
    """
    V = len(model["vocab"])
    k = model["smoothing"]
    if prev not in model["transitions"]:
        return -math.log(max(V, 1))
    count_pair = model["transitions"][prev].get(curr, 0)
    count_prev = model["totals"][prev]
    return math.log((count_pair + k) / (count_prev + k * V))


def sequence_log_prob(chord_seq, model):
    """Mean log-prob per transition for one chord sequence.

    Higher (less negative) = transitions are more like real POP909.
    Returns nan if the sequence has fewer than 2 chords.
    """
    labels = [c.label for c in chord_seq]
    if len(labels) < 2:
        return float("nan")
    lps = [_log_prob_pair(a, b, model) for a, b in zip(labels, labels[1:])]
    return float(np.mean(lps))


def plausibility_set(chord_seqs_dict, model):
    """Score a set of predicted chord sequences under the bigram model.

    Inputs:
        chord_seqs_dict -- {song_id -> ChordSeq}
        model           -- output of fit_bigram(real training chords)

    Returns: dict with two keys
        'per_song'  -- {song_id -> mean log-prob per transition}
        'aggregate' -- mean across songs (nan songs ignored)
    """
    per_song = {sid: sequence_log_prob(seq, model)
                for sid, seq in chord_seqs_dict.items()}
    valid = [v for v in per_song.values() if not np.isnan(v)]
    aggregate = float(np.mean(valid)) if valid else float("nan")
    return {"per_song": per_song, "aggregate": aggregate}
