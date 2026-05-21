"""task2_eval.data_split -- the train/val/test split and POP909 file layout.

POP909 layout this module assumes:
    <data_dir>/001/001.mid          the MIDI (melody is the MELODY track)
    <data_dir>/001/chord_midi.txt   the real chord annotation
    ...

The split is stored as plain-text id files (train_ids.txt etc.) -- the
SAME files used by task1_eval, so both tasks evaluate on the same songs.
"""

import warnings
from pathlib import Path

from .chord_io import load_melody, load_chords


def _normalize_id(song_id):
    """POP909 folders are 3-digit zero-padded ('001').

    Accepts '1', 1, or '001' and returns '001'. Non-numeric ids are
    returned unchanged.
    """
    s = str(song_id).strip()
    return s.zfill(3) if s.isdigit() else s


def read_ids(txt_path):
    """Read a split file (e.g. test_ids.txt): one song id per line.

    Blank lines and lines starting with '#' are ignored.
    Returns a list of normalized id strings.
    """
    ids = []
    for line in Path(txt_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(_normalize_id(line))
    return ids


def load_split(split_dir):
    """Load train/val/test id lists from a folder containing
    train_ids.txt, val_ids.txt and test_ids.txt.

    Returns: {"train": [...], "val": [...], "test": [...]}
    """
    split_dir = Path(split_dir)
    return {
        name: read_ids(split_dir / f"{name}_ids.txt")
        for name in ("train", "val", "test")
    }


def pop909_midi_path(data_dir, song_id):
    """Path to a song's MIDI file: <data_dir>/<id>/<id>.mid."""
    sid = _normalize_id(song_id)
    return Path(data_dir) / sid / f"{sid}.mid"


def pop909_chord_path(data_dir, song_id):
    """Path to a song's chord annotation: <data_dir>/<id>/chord_midi.txt."""
    sid = _normalize_id(song_id)
    return Path(data_dir) / sid / "chord_midi.txt"


def filter_available(data_dir, ids):
    """Keep only the ids whose MIDI file AND chord file both exist.

    Useful when testing with a partial download (e.g. only folders 001
    and 002) against a full test-id list.
    """
    out = []
    for sid in (_normalize_id(i) for i in ids):
        if (pop909_midi_path(data_dir, sid).is_file()
                and pop909_chord_path(data_dir, sid).is_file()):
            out.append(sid)
    return out


def load_song(data_dir, song_id, track_name="MELODY"):
    """Load one POP909 song as (melody, real_chords).

    melody      -- list[Note]   the model's input
    real_chords -- list[Chord]  POP909's ground-truth chord annotation
    """
    sid = _normalize_id(song_id)
    melody = load_melody(pop909_midi_path(data_dir, sid), track_name)
    chords = load_chords(pop909_chord_path(data_dir, sid))
    return melody, chords


def load_songs_by_id(data_dir, ids, track_name="MELODY"):
    """Load several POP909 songs.

    Returns: dict {song_id -> (melody, real_chords)}.

    A dict keyed by id makes it easy to line each song up with the
    matching predicted chords later. Songs that fail to load are skipped
    with a warning rather than crashing the whole run.
    """
    songs = {}
    skipped = []
    for sid in (_normalize_id(i) for i in ids):
        try:
            songs[sid] = load_song(data_dir, sid, track_name)
        except Exception as err:  # keep going through the rest of the batch
            skipped.append((sid, repr(err)))
    if skipped:
        msg = f"skipped {len(skipped)} song(s) that failed to load:\n"
        msg += "\n".join(f"  {sid}: {err}" for sid, err in skipped)
        warnings.warn(msg)
    return songs
