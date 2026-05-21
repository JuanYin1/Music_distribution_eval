"""task2_eval.agreement -- score predicted chords against real chords.

Uses mir_eval.chord, the standard tool for chord-recognition evaluation.

This is the "agreement with the reference harmonization" metric from the
eval plan. We deliberately do NOT call it 'accuracy', because a chord
that differs from the reference can still be musically valid -- that
case is covered separately by chord-melody fit (harmonicity.py).

Each score is reported at several strictness levels at once, so a
near-miss (e.g. predicting C:maj when the real chord is C:maj7) is not
treated the same as a total miss:

    root      -- the chord root only
    majmin    -- root + major/minor distinction
    triads    -- root + triad quality (maj/min/dim/aug/sus)
    sevenths  -- root + 7th chord quality
    tetrads   -- root + full 4-note quality
    mirex     -- MIREX 'good enough' criterion (>= 3 shared chord tones)

All scores live in [0, 1] and are duration-weighted within a song
(longer chord segments count more). Across songs we aggregate by
duration too, so longer songs have proportionally more weight.
"""

import numpy as np
import mir_eval


# Scoring levels we care about, from loose to strict.
SCORE_NAMES = ("root", "majmin", "triads", "sevenths", "tetrads", "mirex")


def _intervals_labels(chord_seq):
    """Convert our ChordSeq into the (intervals, labels) format mir_eval expects."""
    intervals = np.array([[c.start, c.end] for c in chord_seq], dtype=float)
    labels = [c.label for c in chord_seq]
    return intervals, labels


def _song_duration(chord_seq):
    return (chord_seq[-1].end - chord_seq[0].start) if chord_seq else 0.0


def score_song(real_chords, predicted_chords):
    """Score one song's predicted chords against its real chords.

    Returns: dict mapping each strictness level in SCORE_NAMES to a
    score in [0, 1].
    """
    ref_iv, ref_lab = _intervals_labels(real_chords)
    est_iv, est_lab = _intervals_labels(predicted_chords)
    # mir_eval trims to the common time range and merges the two grids.
    full = mir_eval.chord.evaluate(ref_iv, ref_lab, est_iv, est_lab)
    return {name: float(full[name]) for name in SCORE_NAMES if name in full}


def score_set(real_dict, predicted_dict, weight_by_duration=True):
    """Score a whole set of songs.

    Inputs:
        real_dict      -- {song_id -> real ChordSeq}
        predicted_dict -- {song_id -> predicted ChordSeq}
                          Only ids present in BOTH dicts are scored.
        weight_by_duration -- True (default): longer songs count more
                              when aggregating across songs. False:
                              simple mean across songs.

    Returns: dict with two keys
        'per_song'  -- {song_id -> {score_name -> value}}
        'aggregate' -- {score_name -> value} averaged across songs
    """
    common_ids = sorted(set(real_dict) & set(predicted_dict))
    per_song = {sid: score_song(real_dict[sid], predicted_dict[sid])
                for sid in common_ids}
    weights = {
        sid: (_song_duration(real_dict[sid]) if weight_by_duration else 1.0)
        for sid in common_ids
    }

    aggregate = {}
    for name in SCORE_NAMES:
        num = 0.0
        den = 0.0
        for sid in common_ids:
            v = per_song[sid].get(name, np.nan)
            if not np.isnan(v):
                num += v * weights[sid]
                den += weights[sid]
        aggregate[name] = (num / den) if den > 0 else float("nan")

    return {"per_song": per_song, "aggregate": aggregate}
