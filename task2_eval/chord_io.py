"""task2_eval.chord_io -- read melodies and chord sequences for Task 2.

Task 2 compares predicted chords against real chords, so this module
loads both:
  * the melody (the model's input)  -- from the POP909 MIDI file
  * the real chord sequence         -- from POP909's chord_midi.txt
  * a predicted chord sequence      -- from a file, or from a grid of labels

Everything here uses SECONDS as the time unit, so melody notes and chord
segments share one timeline (no tempo conversion needed).

Formats:
    Note     = (pitch, start, end)    pitch: MIDI int; start/end: seconds
    Chord    = (start, end, label)    label: Harte style, e.g. 'C:maj', 'N'
    Melody   = list[Note]  sorted by start
    ChordSeq = list[Chord] sorted by start
"""

from collections import namedtuple
from pathlib import Path

import pretty_midi

Note = namedtuple("Note", ["pitch", "start", "end"])
Chord = namedtuple("Chord", ["start", "end", "label"])

MELODY_TRACK_NAME = "MELODY"


def _pick_track(pm, track_name):
    """Choose which track to read notes from: a track named `track_name`,
    or the single non-empty track if there is exactly one."""
    named = [
        inst for inst in pm.instruments
        if inst.name.strip().upper() == track_name.upper()
    ]
    if named:
        return named[0]
    non_empty = [inst for inst in pm.instruments if inst.notes]
    if len(non_empty) == 1:
        return non_empty[0]
    raise ValueError(
        f"Could not find a '{track_name}' track, and the file has "
        f"{len(non_empty)} non-empty tracks. Pass track_name explicitly."
    )


def load_melody(midi_path, track_name=MELODY_TRACK_NAME):
    """Load a melody from a MIDI file as a list[Note], times in seconds."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    inst = _pick_track(pm, track_name)
    melody = [Note(n.pitch, n.start, n.end) for n in inst.notes]
    melody.sort(key=lambda x: (x.start, x.pitch))
    return melody


def load_chords(chord_txt_path):
    """Load a chord sequence from a POP909-style chord file.

    Each line is:  start_time <whitespace> end_time <whitespace> label
    Times in seconds; labels in Harte style ('C:maj', 'A:min7', 'N').
    Blank lines are ignored.

    This works for POP909's chord_midi.txt AND for any predicted-chord
    file written in the same 3-column format -- which is the recommended
    way for the modeling team to hand over their output.

    Returns: list[Chord] sorted by start.
    """
    chords = []
    for line in Path(chord_txt_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        chords.append(Chord(float(parts[0]), float(parts[1]), parts[2]))
    chords.sort(key=lambda c: c.start)
    return chords


def chords_from_grid(labels, step, start=0.0):
    """Build a ChordSeq from a flat list of chord labels on a fixed grid.

    Use this when a model emits one chord label per slot (e.g. per bar):
    `step` is the slot length in seconds, and slot i covers
    [start + i*step, start + (i+1)*step).

    Returns: list[Chord].
    """
    return [
        Chord(start + i * step, start + (i + 1) * step, label)
        for i, label in enumerate(labels)
    ]
