# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    tiled = np.tile(np.vstack(x), degree+1)
    poly = np.power(tiled, np.arange(degree + 1))
    return poly
