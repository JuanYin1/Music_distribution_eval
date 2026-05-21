"""task1_eval.copying -- check whether the model is memorizing training data.

A model that copies the training set would score a near-perfect
distribution match (see distribution.py) but is not really "generating".
This module measures n-gram overlap: how many of a generated melody's
short note sequences also appear, exactly, somewhere in the training set.

Read the result together with the distribution distances:
    good model  -> low distribution distance AND low copying overlap
    memorizing  -> low distribution distance BUT high copying overlap
"""

import numpy as np


def _to_tokens(melody, transposition_invariant=False):
    """Turn a melody into a token sequence for n-gram comparison.

    By default the tokens are absolute pitches (catches exact copies).
    With transposition_invariant=True the tokens are pitch intervals
    between consecutive notes, which also catches copies shifted into a
    different key.
    """
    pitches = [n.pitch for n in melody]
    if not transposition_invariant:
        return pitches
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]


def _ngrams(tokens, n):
    """All length-n contiguous slices of a token list, as a set of tuples."""
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _train_ngram_pool(train, n, transposition_invariant):
    """One big set of every n-gram appearing anywhere in the training set."""
    pool = set()
    for melody in train:
        pool |= _ngrams(_to_tokens(melody, transposition_invariant), n)
    return pool


def overlap_per_melody(generated, train, n=4, transposition_invariant=False):
    """For each generated melody, the fraction of its unique n-grams that
    also appear somewhere in the training set.

    Melodies too short to contain an n-gram are skipped. Useful for
    finding the single worst-offending (most-copied) generated melody.

    Returns: list[float], one value per (long-enough) generated melody.
    """
    pool = _train_ngram_pool(train, n, transposition_invariant)
    fractions = []
    for melody in generated:
        grams = _ngrams(_to_tokens(melody, transposition_invariant), n)
        if not grams:
            continue
        copied = sum(1 for g in grams if g in pool)
        fractions.append(copied / len(grams))
    return fractions


def ngram_overlap(generated, train, n=4, transposition_invariant=False):
    """Average n-gram overlap between the generated set and the training set.

    0.0 = no generated phrase of length n appears in training (fully
    original); 1.0 = every generated phrase is found in training (pure
    copying). Genuine generation sits low but not exactly zero, because
    short common phrases occur by chance.

    Returns: float (mean over generated melodies), or nan if none are
    long enough.
    """
    fractions = overlap_per_melody(generated, train, n, transposition_invariant)
    return float(np.mean(fractions)) if fractions else float("nan")


def overlap_curve(generated, train, ns=(3, 4, 5, 6, 8),
                  transposition_invariant=False):
    """n-gram overlap measured at several values of n.

    Longer n-grams matching is much stronger evidence of copying: a
    3-note match is often coincidence, an 8-note exact match is not.

    Returns: dict n -> mean overlap.
    """
    return {
        n: ngram_overlap(generated, train, n, transposition_invariant)
        for n in ns
    }
