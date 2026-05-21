"""task2_eval.dataset -- load the preprocessed Task 2 data.

The Task 2 model works on PREPROCESSED data (built in the modeling
notebook): each song is a sequence of bars, where every bar has

    X -- a 12-dim melody pitch-class histogram (sums to 1)
    y -- one coarse chord label, an integer index into a 24-chord
         vocabulary (12 roots x {maj, min})

The notebook saves this as train/val/test_dataset.pkl (a list of
{song_id, X, y}) and chord_vocab.pkl ({chord_to_idx, idx_to_chord}).

This module loads those files and turns the integer chord indices into
label strings ('C:maj', 'A:min') that the metric modules understand.
"""

import pickle


def load_pickle(path):
    """Load a pickled object (a *_dataset.pkl or chord_vocab.pkl file)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def labels_from_indices(y_indices, idx_to_chord):
    """Turn a sequence of chord indices into a list of label strings."""
    return [idx_to_chord[int(i)] for i in y_indices]
