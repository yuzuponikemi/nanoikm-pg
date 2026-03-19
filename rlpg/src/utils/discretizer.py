"""
State Discretizer

Converts continuous state vectors into discrete bin indices,
enabling tabular reinforcement learning methods (e.g., Q-learning)
to work with continuous state spaces.

The InvertedPendulum state is [x, x_dot, theta, theta_dot].
Each dimension is divided into bins using numpy's digitize.
"""

import numpy as np
from typing import List, Tuple, Optional


class StateDiscretizer:
    """
    Converts continuous state vectors into a single integer index.

    Uses uniform binning per dimension, then encodes the multi-index
    into a flat integer via np.ravel_multi_index.

    Parameters
    ----------
    bins_per_dim : list[int]
        Number of bins for each state dimension.
        Default: [10, 10, 20, 10] for [x, x_dot, theta, theta_dot].
        theta uses more bins because it's the most critical variable.
    state_bounds : list[tuple[float, float]]
        (low, high) bounds for each dimension.
        Default: [(-2.4, 2.4), (-3.0, 3.0), (-0.2095, 0.2095), (-3.0, 3.0)]
        The theta bounds match the environment's theta_threshold (~12 degrees).

    Examples
    --------
    >>> disc = StateDiscretizer()
    >>> state = np.array([0.0, 0.0, 0.05, 0.1])
    >>> idx = disc.encode(state)
    >>> center = disc.decode(idx)
    """

    DEFAULT_BINS = [10, 10, 20, 10]
    DEFAULT_BOUNDS = [
        (-2.4, 2.4),        # x: cart position
        (-3.0, 3.0),        # x_dot: cart velocity
        (-0.2095, 0.2095),  # theta: pole angle (~12 degrees, matches theta_threshold)
        (-3.0, 3.0),        # theta_dot: pole angular velocity
    ]

    def __init__(
        self,
        bins_per_dim: Optional[List[int]] = None,
        state_bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        self.bins_per_dim = bins_per_dim or self.DEFAULT_BINS
        self.state_bounds = state_bounds or self.DEFAULT_BOUNDS

        assert len(self.bins_per_dim) == len(self.state_bounds), (
            "bins_per_dim and state_bounds must have the same length"
        )

        # Precompute bin edges for each dimension
        self._edges = [
            np.linspace(lo, hi, n + 1)
            for (lo, hi), n in zip(self.state_bounds, self.bins_per_dim)
        ]

        self.n_dims = len(self.bins_per_dim)
        self.n_states = int(np.prod(self.bins_per_dim))

    def encode(self, state: np.ndarray) -> int:
        """
        Convert a continuous state vector to a flat integer index.

        Values outside the bounds are clipped to the nearest bin.

        Parameters
        ----------
        state : np.ndarray, shape (n_dims,)

        Returns
        -------
        int : 0 <= idx < self.n_states
        """
        indices = []
        for i, (val, edges) in enumerate(zip(state, self._edges)):
            idx = int(np.digitize(float(val), edges)) - 1
            idx = int(np.clip(idx, 0, self.bins_per_dim[i] - 1))
            indices.append(idx)
        return int(np.ravel_multi_index(indices, self.bins_per_dim))

    def decode(self, idx: int) -> np.ndarray:
        """
        Convert a flat index back to the bin center values.

        Useful for visualization and debugging.

        Parameters
        ----------
        idx : int

        Returns
        -------
        np.ndarray, shape (n_dims,) : center of each bin
        """
        multi_idx = np.unravel_index(idx, self.bins_per_dim)
        centers = []
        for i, (mi, edges) in enumerate(zip(multi_idx, self._edges)):
            center = float((edges[mi] + edges[mi + 1]) / 2.0)
            centers.append(center)
        return np.array(centers)

    def encode_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Encode a batch of states.

        Parameters
        ----------
        states : np.ndarray, shape (N, n_dims)

        Returns
        -------
        np.ndarray of int, shape (N,)
        """
        return np.array([self.encode(s) for s in states])

    def __repr__(self) -> str:
        return (
            f"StateDiscretizer("
            f"bins={self.bins_per_dim}, "
            f"n_states={self.n_states})"
        )
