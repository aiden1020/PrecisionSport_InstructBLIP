import random
import numpy as np
from typing import List, Tuple, Callable, Optional

class VideoDiversitySampler:
    """
    Video diversity sampler supporting multiple distance metrics.

    Args:
        distance_fn: function(a, b) -> float, distance metric. Defaults to cosine distance.
    """
    def __init__(
        self,
        distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    ):
        # Default to cosine distance if none provided
        self.distance_fn = distance_fn if distance_fn is not None else self._cosine_dist


    def _cosine_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _euclidean_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _manhattan_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(np.abs(a - b)))

    def _correlation_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        # 1 - Pearson correlation coefficient
        corr = np.corrcoef(a, b)[0, 1]
        return 1.0 - float(corr)

    def _farthest_first(
        self,
        features: List[np.ndarray],
        M: int
    ) -> Tuple[List[int], List[float]]:
        """
        Farthest-first sampling to select M most diverse items by maximizing minimum distance.

        Args:
            features: list of feature vectors (ndarray), possibly multi-dim
            M:        number of items to select
        Returns:
            ordering: indices of selected items in selection order
            scores:   corresponding minimum distance at each selection step
        """
        n = len(features)
        if n == 0 or M <= 0:
            return [], []

        # Preprocess: flatten features to 1D
        proc_feats = []
        for f in features:
            arr = np.asarray(f)
            if arr.ndim > 1:
                arr = arr.flatten()
            proc_feats.append(arr)

        # 1) Initialize: pick a random seed
        seed_idx = random.randrange(n)
        ordering = [seed_idx]

        # 2) Compute initial min distances from seed
        min_dists = [float('inf')] * n
        for i in range(n):
            if i != seed_idx:
                min_dists[i] = self.distance_fn(proc_feats[i], proc_feats[seed_idx])

        # Record seed score as 0.0 (placeholder)
        scores = [0.0]
        unselected = set(range(n)) - {seed_idx}

        # 3) Iteratively select the farthest-first
        while len(ordering) < M and unselected:
            # pick index with maximum min_dist
            best_idx = max(unselected, key=lambda i: min_dists[i])
            best_score = min_dists[best_idx]

            ordering.append(best_idx)
            scores.append(best_score)
            unselected.remove(best_idx)

            # Update min distances for unselected points
            for j in list(unselected):
                d = self.distance_fn(proc_feats[j], proc_feats[best_idx])
                if d < min_dists[j]:
                    min_dists[j] = d

        return ordering, scores

    def run(
        self,
        M: int,
        features: List[np.ndarray]
    ) -> Tuple[List[int], List[float]]:
        """
        Select M diverse samples from the feature list.

        Args:
            M: number of items to select
            features: list of feature vectors
        Returns:
            ordering, scores
        """
        return self._farthest_first(features, M)
