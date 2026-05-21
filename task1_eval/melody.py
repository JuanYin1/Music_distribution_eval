"""task1_eval.melody -- the internal melody format and a builder for it.

A melody is a list of Note(pitch, start, end). The HMM model and the
POP909 dataset both describe notes by duration, so melodies are built
back-to-back (note i starts where note i-1 ended, no rests). Building
the real and generated melodies the SAME way keeps the distribution
comparison fair.
"""

from collections import namedtuple

# A single note. A "Melody" is a list[Note] sorted by start time.
Note = namedtuple("Note", ["pitch", "start", "end"])


def melody_from_durations(pitch_duration_pairs):
    """Build a Melody from (pitch, duration) pairs, placed back-to-back.

    `pitch_duration_pairs` is an iterable of (pitch, duration). Notes are
    laid out with no gaps: note i starts where note i-1 ended. Use this
    for BOTH the generated melodies and the real POP909 melodies so the
    two sets are constructed identically.

    Returns: list[Note].
    """
    melody = []
    t = 0.0
    for pitch, duration in pitch_duration_pairs:
        d = float(duration)
        melody.append(Note(int(pitch), t, t + d))
        t += d
    return melody
