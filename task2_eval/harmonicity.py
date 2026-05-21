"""task2_eval.harmonicity -- chord-and-melody fit.

For each melody note, ask: 'is this note a chord tone of the predicted
chord at this moment?' The fraction of yes-answers (duration-weighted by
default) is the harmonicity score.

This is the 'good but different' metric from the eval plan: a predicted
chord may differ from POP909's reference and still get a high
harmonicity score, because it can still fit the melody.

Notes that span a chord boundary are assigned to the chord active at
their start time -- a clean approximation for POP909 (chord changes
mostly land on bar boundaries, and so do most melody notes).
"""

import bisect

import numpy as np

from . import chords


def _chord_at(chord_seq, t, starts):
    """Return the Chord active at time t, or None if t is outside the
    chord sequence."""
    idx = bisect.bisect_right(starts, t) - 1
    if 0 <= idx < len(chord_seq):
        c = chord_seq[idx]
        if c.start <= t < c.end:
            return c
    return None


def harmonicity_score(melody, chord_seq,
                      weight_by_duration=True, skip_no_chord=True):
    """How well the chord sequence fits the melody.

    Inputs:
        melody             -- list[Note]
        chord_seq          -- list[Chord]  (typically the predicted chords)
        weight_by_duration -- True: each note weighted by its length;
                              False: each note weighted equally.
        skip_no_chord      -- True: melody notes during 'N' segments are
                              skipped (the model said 'no chord here',
                              so we don't penalize it). False: count
                              them as non-chord tones.

    Returns a float in [0, 1], or nan if no melody notes overlap a
    valid chord.
    """
    if not melody or not chord_seq:
        return float("nan")
    starts = [c.start for c in chord_seq]

    matched = 0.0
    total = 0.0
    for note in melody:
        c = _chord_at(chord_seq, note.start, starts)
        if c is None:
            continue
        if chords.is_no_chord(c.label):
            if skip_no_chord:
                continue
            chord_pcs = set()
        else:
            chord_pcs = chords.pitch_classes(c.label)
            if not chord_pcs:        # unparseable label -- skip rather than punish
                continue

        weight = (note.end - note.start) if weight_by_duration else 1.0
        if (note.pitch % 12) in chord_pcs:
            matched += weight
        total += weight

    return float(matched / total) if total > 0 else float("nan")


def harmonicity_set(songs_with_pred,
                    weight_by_duration=True, skip_no_chord=True):
    """Harmonicity scored over a set of songs.

    `songs_with_pred` is a dict {song_id -> (melody, predicted_chord_seq)}.

    Returns: dict with two keys
        'per_song'  -- {song_id -> score}
        'aggregate' -- mean across songs (nan songs ignored)
    """
    per_song = {
        sid: harmonicity_score(melody, pred,
                               weight_by_duration=weight_by_duration,
                               skip_no_chord=skip_no_chord)
        for sid, (melody, pred) in songs_with_pred.items()
    }
    valid = [v for v in per_song.values() if not np.isnan(v)]
    aggregate = float(np.mean(valid)) if valid else float("nan")
    return {"per_song": per_song, "aggregate": aggregate}
