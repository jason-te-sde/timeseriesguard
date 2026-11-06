"""An Isolation Forest detector over the standardized feature matrix.

It wraps scikit-learn's forest with the three things the protocol needs: scores that rise with
anomaly, a state digest that does not depend on thread count or memory addresses, and a
sample size that is clipped to the rows it was given.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from tsguard.models.base import InputKind, check_context
from tsguard.repro.hashing import array_digest, json_digest


class IsolationForestDetector:
    """Score a row by how quickly random splits isolate it from the training rows.

    The score is the negated scikit-learn ``score_samples``, so a higher score is more
    anomalous. Each row is scored on its own, so no context is needed.

    Attributes:
        name: ``"iforest"``.
        input_kind: ``"features"``.
        lookback: ``1``; a score reads only its own row.
        n_estimators: Number of trees.
        max_samples: Rows drawn for each tree, before clipping to the rows given to ``fit``.
        max_features: Fraction of the columns each tree sees; 1.0 gives every tree all of them.
        seed: Seed of the forest.
        n_jobs: Threads used to fit the forest.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_samples: int = 1024,
        max_features: float = 1.0,
        seed: int,
        n_jobs: int = 4,
    ) -> None:
        """Create an unfitted detector.

        Args:
            n_estimators: Number of trees. Must be at least 1.
            max_samples: Rows drawn for each tree. Must be at least 1; ``fit`` clips it to the
                number of rows it receives.
            max_features: Fraction of the columns each tree sees, in ``(0, 1]``. The default of
                1.0 turns feature bagging off: an Isolation Forest already draws a random feature
                at every split, and a smaller fraction makes scikit-learn copy a column subset
                for every tree (on 24,000 rows by 2,246 columns with 200 trees, fitting took
                0.2 s at 1.0 and 14.7 s at 0.5).
            seed: Seed of the forest. The same seed on the same rows gives the same trees,
                whatever ``n_jobs`` is.
            n_jobs: Threads used to fit. Must be at least 1.

        Raises:
            ValueError: A count is below 1 or ``max_features`` is outside ``(0, 1]``.
        """
        if n_estimators < 1:
            raise ValueError(f"iforest n_estimators must be at least 1, got {n_estimators}")
        if max_samples < 1:
            raise ValueError(f"iforest max_samples must be at least 1, got {max_samples}")
        if not 0.0 < max_features <= 1.0:
            raise ValueError(f"iforest max_features must be in (0, 1], got {max_features}")
        _check_n_jobs(n_jobs)
        self.name: str = "iforest"
        self.input_kind: InputKind = "features"
        self.lookback: int = 1
        self.n_estimators: int = n_estimators
        self.max_samples: int = max_samples
        self.max_features: float = max_features
        self.seed: int = seed
        self.n_jobs: int = n_jobs
        self._model: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> None:
        """Grow the forest on ``X``.

        Args:
            X: Array of shape ``(n_rows, n_columns)`` with at least one row.

        Raises:
            ValueError: ``X`` is not a non-empty 2-D array of finite values.
        """
        _check_matrix(X)
        if len(X) == 0:
            raise ValueError("iforest cannot fit on zero rows")
        model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(X)),
            max_features=self.max_features,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        model.fit(X)
        self._model = model

    def set_n_jobs(self, n: int) -> None:
        """Change the thread count, for a detector that is fitted and now only scores.

        Serving sets 1 so that one request never fans out across the machine. The thread count
        does not change any score or the state digest.

        Args:
            n: Number of threads. Must be at least 1.

        Raises:
            ValueError: ``n < 1``.
        """
        _check_n_jobs(n)
        self.n_jobs = n
        if self._model is not None:
            self._model.n_jobs = n

    def score(self, X: np.ndarray, *, context: int) -> np.ndarray:
        """Score every row of ``X`` after its first ``context`` rows.

        Args:
            X: Array of shape ``(n_rows, n_columns)``, float32 or float64.
            context: Number of leading rows to skip. Any value from 0 up to ``len(X) - 1``.

        Returns:
            A float64 array of ``len(X) - context`` scores; higher is more anomalous.

        Raises:
            ValueError: ``context`` is negative or leaves no rows, the column count differs
                from the fitted one, or the scored rows hold NaN or infinity.
            RuntimeError: The detector is not fitted.
        """
        check_context(self, X, context)
        if self._model is None:
            raise RuntimeError("iforest is not fitted: call fit before score")
        rows = X[context:]
        _check_matrix(rows)
        if rows.shape[1] != self._model.n_features_in_:
            raise ValueError(
                f"iforest was fitted on {self._model.n_features_in_} columns, "
                f"got {rows.shape[1]} columns"
            )
        return np.asarray(-self._model.score_samples(rows), dtype=np.float64)

    def state_digest(self) -> str:
        """Return a digest of every tree and of the sample size that normalizes the scores.

        For each tree, in estimator order, the digest covers the split features and thresholds,
        the number of training rows that reached each node (a row's path length is corrected by
        the rows left in its leaf, so two forests with the same splits can still score
        differently), and the feature subset the tree was given.

        Returns:
            A 64-character hex string, equal for two detectors fitted on the same rows with the
            same configuration and seed, whatever their thread counts.

        Raises:
            RuntimeError: The detector is not fitted.
        """
        if self._model is None:
            raise RuntimeError("iforest is not fitted: call fit before state_digest")
        trees = [
            [
                array_digest(tree.tree_.feature),
                array_digest(tree.tree_.threshold),
                array_digest(tree.tree_.n_node_samples),
                array_digest(np.asarray(subset)),
            ]
            for tree, subset in zip(
                self._model.estimators_, self._model.estimators_features_, strict=True
            )
        ]
        return json_digest({"max_samples": int(self._model.max_samples_), "trees": trees})


def _check_n_jobs(n_jobs: int) -> None:
    """Raise ValueError unless ``n_jobs`` is at least 1."""
    if n_jobs < 1:
        raise ValueError(f"iforest n_jobs must be at least 1, got {n_jobs}")


def _check_matrix(X: np.ndarray) -> None:
    """Raise ValueError unless ``X`` is a 2-D array of finite values."""
    if X.ndim != 2:
        raise ValueError(f"iforest needs a 2-D array of rows and columns, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("iforest needs finite values: X holds NaN or infinity")
