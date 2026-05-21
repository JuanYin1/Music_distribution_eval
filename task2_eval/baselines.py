"""task2_eval.baselines -- simple chord-prediction baselines.

Three baselines from the eval plan:
  1. most_common_baseline      -- always predict the single most-frequent
                                  chord from the training set.
  2. rule_based_baseline       -- for each segment, pick the major/minor
                                  triad that contains the most melody
                                  notes during that segment.
  3. melody_ignoring_baseline  -- sample a chord sequence from a bigram
                                  model trained on real chord transitions,
                                  IGNORING the melody.

A good model should beat all three. Beating (3) is especially important:
it proves the model is actually using the melody it was given -- if the
model can't beat a chord-only model, the melody is being ignored.

Each baseline returns a ChordSeq with the same time intervals as the
template you pass in (usually the real chord sequence), so the output
can be compared directly with the agreement and harmonicity metrics.
"""

import random
from collections import Counter

from .chord_io import Chord
from .chords import pitch_classes
from .progression import fit_bigram


# Major and minor triads at every root: 24 candidates for the rule-based pick.
_PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_TRIAD_TEMPLATES = [(root, q) for root in range(12) for q in ("maj", "min")]


def _label(root, quality):
    return f"{_PC_NAMES[root]}:{quality}"


def _unigram(training_chord_seqs):
    """Count how often each label appears in the training data."""
    counter = Counter()
    for seq in training_chord_seqs:
        for c in seq:
            counter[c.label] += 1
    return counter


def _sample_bigram_sequence(length, bigram, unigram, rng):
    """Sample `length` labels from a bigram model, starting from the
    unigram and continuing by `prev -> next` transition counts."""
    if length <= 0:
        return []
    if unigram:
        u_labels, u_weights = zip(*unigram.items())
    else:
        u_labels, u_weights = ("N",), (1,)
    out = [rng.choices(u_labels, weights=u_weights, k=1)[0]]
    for _ in range(1, length):
        prev = out[-1]
        trans = bigram["transitions"].get(prev)
        if trans:
            ks = list(trans.keys())
            ws = list(trans.values())
            out.append(rng.choices(ks, weights=ws, k=1)[0])
        else:
            out.append(rng.choices(u_labels, weights=u_weights, k=1)[0])
    return out


# ---------- 1. Most-common chord ---------------------------------------

def most_common_baseline(template_chord_seq, training_chord_seqs):
    """Always predict the single most-frequent training label.

    A floor: shows how high agreement can go just by exploiting class
    imbalance (e.g. many songs sit on the tonic chord a lot).
    """
    counter = _unigram(training_chord_seqs)
    top = counter.most_common(1)[0][0] if counter else "N"
    return [Chord(c.start, c.end, top) for c in template_chord_seq]


# ---------- 2. Rule-based diatonic ------------------------------------

def _melody_pcs_in_window(melody, start, end):
    """Pitch classes of every melody note that overlaps [start, end)."""
    return [n.pitch % 12 for n in melody if n.end > start and n.start < end]


def _best_triad(melody_pcs):
    """Pick the maj/min triad that contains the most melody pitch classes."""
    if not melody_pcs:
        return None
    pc_counter = Counter(melody_pcs)
    best_label = None
    best_score = -1
    for root, quality in _TRIAD_TEMPLATES:
        chord_pcs = pitch_classes(_label(root, quality))
        score = sum(pc_counter[pc] for pc in chord_pcs)
        if score > best_score:
            best_score = score
            best_label = _label(root, quality)
    return best_label


def rule_based_baseline(template_chord_seq, melody):
    """For each template segment, pick the maj/min triad with the most
    melody pitch classes in that window.

    A real music-theory baseline: far smarter than 'always C:maj' but
    knows nothing about chord progressions or 7ths.
    """
    out = []
    for c in template_chord_seq:
        mel_pcs = _melody_pcs_in_window(melody, c.start, c.end)
        out.append(Chord(c.start, c.end, _best_triad(mel_pcs) or "N"))
    return out


# ---------- 3. Melody-ignoring chord bigram ---------------------------

def melody_ignoring_baseline(template_chord_seq, training_chord_seqs, seed=0):
    """Sample a chord sequence from a bigram trained on real transitions,
    completely ignoring the melody.

    Beating this baseline proves your model is actually using the melody
    it was given. If a chord-only model can match your scores, the
    melody input is doing nothing.
    """
    bigram = fit_bigram(training_chord_seqs)
    unigram = _unigram(training_chord_seqs)
    rng = random.Random(seed)
    labels = _sample_bigram_sequence(len(template_chord_seq),
                                     bigram, unigram, rng)
    return [Chord(c.start, c.end, lab)
            for c, lab in zip(template_chord_seq, labels)]


# ---------- Convenience: run all three on a whole test set -----------

def build_baselines(test_songs, training_chord_seqs, seed=0):
    """Produce baseline predictions for a whole test set at once.

    Inputs:
        test_songs          -- {song_id -> (melody, real_chords)}
                               real_chords are used as the time-grid template.
        training_chord_seqs -- iterable of real training ChordSeqs.

    Returns: {baseline_name -> {song_id -> ChordSeq}}, with keys
    'most_common', 'rule_based', 'melody_ignoring'. Each inner dict
    plugs straight into agreement.score_set / harmonicity.harmonicity_set.
    """
    bigram = fit_bigram(training_chord_seqs)
    unigram = _unigram(training_chord_seqs)
    top_label = unigram.most_common(1)[0][0] if unigram else "N"
    rng = random.Random(seed)

    out = {"most_common": {}, "rule_based": {}, "melody_ignoring": {}}
    for sid, (mel, template) in test_songs.items():
        out["most_common"][sid] = [
            Chord(c.start, c.end, top_label) for c in template
        ]
        out["rule_based"][sid] = rule_based_baseline(template, mel)
        labels = _sample_bigram_sequence(len(template), bigram, unigram, rng)
        out["melody_ignoring"][sid] = [
            Chord(c.start, c.end, lab) for c, lab in zip(template, labels)
        ]
    return out
