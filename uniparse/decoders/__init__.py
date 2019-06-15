"""decoder module."""
from uniparse.decoders.eisner import Eisner as _eisner
from uniparse.decoders.cle import parse_nonproj as _cle

import numpy as np


def _decode(alg, scores, clip):
    if clip is None:
        return np.array([alg(s) for s in scores])

    return np.array([alg(s[:l, :l]) for s, l in zip(scores, clip)])


def eisner(scores, clip=None):
    """
    Eisner algorithm on single square matrix.

    Assumes batches along 0 dimension
    """
    return _decode(_eisner, scores, clip)


def cle(scores, clip=None):
    """
    Chu-liu-edmonds algorithm on single square matrix.

    Assumes batches along 0 dimension
    """
    return _decode(_cle, scores, clip)
