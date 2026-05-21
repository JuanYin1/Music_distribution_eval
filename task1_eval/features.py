"""task1_eval.features -- turn a melody into feature numbers.

Each feature function takes a Melody (list[Note], times in beats) and
returns a single float. Across a set of melodies, each feature becomes a
distribution we can compare (see distribution.py).

Empty melodies, or features undefined for a melody (e.g. the interval of
a one-note melody), return float('nan').
"""

from collections import Counter
import math

import numpy as np

BEATS_PER_BAR = 4.0  # POP909 is almost entirely in 4/4 time.

# Pitch-class offsets for the major and natural-minor scales.
_MAJOR = {0, 2, 4, 5, 7, 9, 11}
_MINOR = {0, 2, 3, 5, 7, 8, 10}


def _span_in_beats(melody):
    """Length of the melody, from the first note's start to the last
    note's end, in beats."""
    return max(n.end for n in melody) - melody[0].start


# --- individual features --------------------------------------------------

def pitch_range(melody):
    """Highest pitch minus lowest pitch, in semitones."""
    if not melody:
        return float("nan")
    pitches = [n.pitch for n in melody]
    return float(max(pitches) - min(pitches))


def mean_pitch(melody):
    """Average MIDI pitch."""
    if not melody:
        return float("nan")
    return float(np.mean([n.pitch for n in melody]))


def note_density(melody):
    """Average number of notes per bar (bar = 4 beats)."""
    if not melody:
        return float("nan")
    span = _span_in_beats(melody)
    if span <= 0:
        return float("nan")
    return len(melody) / (span / BEATS_PER_BAR)


def mean_note_duration(melody):
    """Average note length, in beats."""
    if not melody:
        return float("nan")
    return float(np.mean([n.end - n.start for n in melody]))


def mean_abs_interval(melody):
    """Average absolute pitch jump between consecutive notes, in semitones.

    Undefined (nan) for melodies with fewer than two notes.
    """
    if len(melody) < 2:
        return float("nan")
    pitches = [n.pitch for n in melody]
    jumps = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    return float(np.mean(jumps))


def pitch_class_histogram(melody):
    """12-bin pitch-class histogram, normalized to sum to 1.

    Returns a numpy array of length 12 (index 0 = C, 1 = C#, ...).
    This one returns an array, not a scalar, so it is kept out of
    extract_all -- it is meant for plotting, not distribution distance.
    """
    hist = np.zeros(12)
    for n in melody:
        hist[n.pitch % 12] += 1
    total = hist.sum()
    return hist if total == 0 else hist / total


def pitch_class_entropy(melody):
    """Entropy of the pitch-class histogram, in bits.

    Low entropy = the melody sticks to a few pitch classes; high entropy
    = it spreads across many. Maximum possible is log2(12) ~= 3.585.
    """
    hist = pitch_class_histogram(melody)
    nonzero = hist[hist > 0]
    if nonzero.size == 0:
        return float("nan")
    return float(-np.sum(nonzero * np.log2(nonzero)))


def scale_consistency(melody):
    """Fraction of notes that fit the best-matching major or minor scale.

    Tries all 12 major and 12 minor scales and returns the highest
    fraction of notes whose pitch class lies in that scale. 1.0 means
    every note fits some single scale.
    """
    if not melody:
        return float("nan")
    counts = Counter(n.pitch % 12 for n in melody)
    total = len(melody)
    best = 0.0
    for root in range(12):
        for scale in (_MAJOR, _MINOR):
            in_scale = sum(
                counts[pc] for pc in range(12) if (pc - root) % 12 in scale
            )
            best = max(best, in_scale / total)
    return best


def empty_bar_rate(melody):
    """Fraction of bars (4-beat windows from beat 0) with no note onset."""
    if not melody:
        return float("nan")
    bars_with_notes = {int(n.start // BEATS_PER_BAR) for n in melody}
    span_end = max(n.end for n in melody)
    n_bars = max(math.ceil(span_end / BEATS_PER_BAR), max(bars_with_notes) + 1)
    return 1.0 - len(bars_with_notes) / n_bars


# --- running all features --------------------------------------------------

# Every scalar feature, in a stable order. (pitch_class_histogram is an
# array, not a scalar, so it is intentionally not here.)
SCALAR_FEATURES = {
    "pitch_range": pitch_range,
    "mean_pitch": mean_pitch,
    "pitch_class_entropy": pitch_class_entropy,
    "scale_consistency": scale_consistency,
    "note_density": note_density,
    "mean_note_duration": mean_note_duration,
    "mean_abs_interval": mean_abs_interval,
    "empty_bar_rate": empty_bar_rate,
}


def extract_all(melody):
    """Run every scalar feature on one melody.

    Returns: dict mapping feature name -> float.
    """
    return {name: func(melody) for name, func in SCALAR_FEATURES.items()}


def extract_set(melodies):
    """Run every scalar feature on a set of melodies.

    Returns: dict mapping feature name -> list of values (one per
    melody). This is the per-feature distribution used by
    distribution.py.
    """
    result = {name: [] for name in SCALAR_FEATURES}
    for melody in melodies:
        for name, value in extract_all(melody).items():
            result[name].append(value)
    return result
