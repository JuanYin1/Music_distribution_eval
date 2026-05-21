"""task2_eval.chords -- chord label utilities.

Wraps mir_eval.chord to:
  * turn a chord label like 'C:maj' or 'A:min7' into the set of pitch
    classes that make up the chord (used for the chord-melody-fit metric)
  * tell whether a label means 'no chord' (POP909 uses 'N')
  * pull the unique chord vocabulary out of a set of songs

Pitch classes are integers 0-11 with C = 0.
"""

import mir_eval


# Labels in POP909 (and most chord datasets) that mean "no chord here".
_NO_CHORD_LABELS = {"N", "X", "NC", ""}


def is_no_chord(label):
    """Whether `label` means 'no chord here'."""
    return label.strip().upper() in _NO_CHORD_LABELS


def pitch_classes(label):
    """The set of pitch classes (0-11, C=0) that make up the chord.

    Uses mir_eval.chord.encode, which handles root + quality + extensions
    + inversions correctly (e.g. 'C:maj7', 'A:min7/3').

    For 'N' (no chord) or an unparseable label, returns the empty set.
    """
    if is_no_chord(label):
        return set()
    try:
        root, bitmap, _bass = mir_eval.chord.encode(label)
    except mir_eval.chord.InvalidChordException:
        return set()
    if root < 0:
        return set()
    return {(root + i) % 12 for i in range(12) if bitmap[i]}


def root_pitch_class(label):
    """The chord's root pitch class (0-11, C=0).

    Returns None for 'N' or an unparseable label.
    """
    if is_no_chord(label):
        return None
    try:
        root, _, _ = mir_eval.chord.encode(label)
    except mir_eval.chord.InvalidChordException:
        return None
    return root if root >= 0 else None


def chord_vocabulary(chord_seqs):
    """All unique chord labels across a collection of ChordSeqs.

    `chord_seqs` is an iterable of list[Chord]. Useful for building the
    progression model (progression.py) and the most-common-chord
    baseline (baselines.py).

    Returns a sorted list of label strings.
    """
    vocab = set()
    for seq in chord_seqs:
        for c in seq:
            vocab.add(c.label)
    return sorted(vocab)
