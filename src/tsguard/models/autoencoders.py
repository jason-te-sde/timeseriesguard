"""Window autoencoders over the raw channels: a 1D-CNN and an LSTM.

Both learn to reconstruct a window of recent rows from a narrow bottleneck. A row is scored by
how badly the model reconstructs it at the end of its window, so a row that breaks the pattern
the model learned from normal data scores high. Training is deterministic on CPU: the weights
are initialized from the seed inside a forked torch RNG, batches are drawn from one seeded
generator, and nothing in the loop reads the global torch RNG.
"""

from __future__ import annotations

from typing import Final, Literal, get_args

import numpy as np
import torch
from torch import nn

from tsguard.models.base import InputKind, check_context
from tsguard.repro.hashing import array_digest, json_digest

Kind = Literal["cnn", "lstm"]

_KERNEL: Final = 5
_SCORE_BATCH: Final = 2048  # windows per forward pass while scoring


class CnnAutoencoder(nn.Module):
    """A four-layer 1D convolutional autoencoder with a ``latent``-channel bottleneck.

    The layers are ``Conv1d(C, width, 5) -> Conv1d(width, latent, 5) -> Conv1d(latent, width, 5)
    -> Conv1d(width, C, 5)``, each padded to keep the window length, with a ReLU between them.
    """

    def __init__(self, n_channels: int, width: int, latent: int) -> None:
        """Create the network with torch's default initialization.

        Args:
            n_channels: Number of input channels ``C``.
            width: Channels of the two outer hidden layers.
            latent: Channels of the bottleneck layer.
        """
        super().__init__()
        pad = _KERNEL // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, width, _KERNEL, padding=pad),
            nn.ReLU(),
            nn.Conv1d(width, latent, _KERNEL, padding=pad),
            nn.ReLU(),
            nn.Conv1d(latent, width, _KERNEL, padding=pad),
            nn.ReLU(),
            nn.Conv1d(width, n_channels, _KERNEL, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of windows.

        Args:
            x: Tensor of shape ``(batch, window, channels)``.

        Returns:
            The reconstruction, with the same shape as ``x``.
        """
        reconstructed: torch.Tensor = self.net(x.permute(0, 2, 1))
        return reconstructed.permute(0, 2, 1)


class LstmAutoencoder(nn.Module):
    """An LSTM autoencoder that squeezes a window into one ``latent`` vector and unrolls it.

    The encoder LSTM reads the window and its last hidden state goes through a linear layer to
    ``latent`` numbers. A second linear layer maps them back to ``hidden`` numbers, which are
    repeated at every step as the decoder LSTM's input; a last linear layer maps each decoder
    output to the channels.
    """

    def __init__(self, n_channels: int, hidden: int, latent: int) -> None:
        """Create the network with torch's default initialization.

        Args:
            n_channels: Number of input channels ``C``.
            hidden: Hidden size of both LSTMs.
            latent: Size of the bottleneck vector.
        """
        super().__init__()
        self.encoder = nn.LSTM(n_channels, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.to_channels = nn.Linear(hidden, n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of windows.

        Args:
            x: Tensor of shape ``(batch, window, channels)``.

        Returns:
            The reconstruction, with the same shape as ``x``.
        """
        _, (last_hidden, _) = self.encoder(x)
        code = self.to_latent(last_hidden[-1])
        seed = self.from_latent(code)
        repeated = seed.unsqueeze(1).expand(-1, x.shape[1], -1)
        decoded, _ = self.decoder(repeated)
        reconstructed: torch.Tensor = self.to_channels(decoded)
        return reconstructed


class AutoencoderDetector:
    """Score a row by the reconstruction error of the last step of the window that ends at it.

    ``fit`` trains on windows of ``window`` consecutive rows taken every ``stride`` rows. ``score``
    slides the window one row at a time and, for each window, averages over the channels the
    squared difference between the reconstruction and the input at the window's last step.

    Attributes:
        name: ``"cnn_ae"`` or ``"lstm_ae"``.
        input_kind: ``"raw"``.
        lookback: ``window``; a score reads its own row and the ``window - 1`` rows before it.
        kind: ``"cnn"`` or ``"lstm"``.
        window: Rows per window.
        latent: Size of the bottleneck.
        size: Width of the CNN's hidden layers, or hidden size of the LSTMs.
        lr: Adam learning rate.
        seed: Seed of the weight initialization and of the batch order.
        epochs: Passes over the training windows.
        stride: Rows between the starts of two training windows.
        batch_size: Training windows per optimizer step.
    """

    def __init__(
        self,
        *,
        kind: Kind,
        window: int,
        latent: int,
        size: int,
        lr: float,
        seed: int,
        epochs: int = 8,
        stride: int = 4,
        batch_size: int = 256,
    ) -> None:
        """Create an unfitted detector.

        Args:
            kind: ``"cnn"`` or ``"lstm"``.
            window: Rows per window. Must be at least 1.
            latent: Size of the bottleneck. Must be at least 1.
            size: Width (CNN) or hidden size (LSTM). Must be at least 1.
            lr: Adam learning rate. Must be positive.
            seed: Seed of the weights and the batch order.
            epochs: Passes over the training windows. Must be at least 1.
            stride: Rows between training windows. Must be at least 1.
            batch_size: Training windows per optimizer step. Must be at least 1.

        Raises:
            ValueError: ``kind`` is not one of the two names, a count is below 1, or ``lr`` is
                not positive.
        """
        if kind not in get_args(Kind):
            raise ValueError(f"autoencoder kind must be one of {get_args(Kind)}, got {kind!r}")
        for label, value in (
            ("window", window),
            ("latent", latent),
            ("size", size),
            ("epochs", epochs),
            ("stride", stride),
            ("batch_size", batch_size),
        ):
            if value < 1:
                raise ValueError(f"{kind}_ae {label} must be at least 1, got {value}")
        if not lr > 0.0:
            raise ValueError(f"{kind}_ae lr must be positive, got {lr}")
        self.name: str = f"{kind}_ae"
        self.input_kind: InputKind = "raw"
        self.lookback: int = window
        self.kind: Kind = kind
        self.window: int = window
        self.latent: int = latent
        self.size: int = size
        self.lr: float = lr
        self.seed: int = seed
        self.epochs: int = epochs
        self.stride: int = stride
        self.batch_size: int = batch_size
        self._model: nn.Module | None = None
        self._n_channels: int = 0

    def fit(self, X: np.ndarray) -> None:
        """Train the autoencoder on windows of ``X``.

        The weights are initialized from ``seed`` inside a forked torch RNG, so the global torch
        RNG state is the same after ``fit`` as before it. Calling ``fit`` again retrains from
        scratch.

        Args:
            X: Array of shape ``(n_rows, n_channels)`` with at least ``window`` rows.

        Raises:
            ValueError: ``X`` is not a 2-D array of finite values, or has fewer than ``window``
                rows.
        """
        data = self._as_matrix(X)
        n_rows, n_channels = data.shape
        if n_rows < self.window:
            raise ValueError(
                f"{self.name} needs at least window={self.window} rows to fit, got {n_rows}"
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            model = self._build(n_channels)
        rows = torch.from_numpy(data)
        starts = torch.arange(0, n_rows - self.window + 1, self.stride)
        offsets = torch.arange(self.window)
        generator = torch.Generator().manual_seed(self.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(starts), generator=generator)
            for first in range(0, len(order), self.batch_size):
                batch_starts = starts[order[first : first + self.batch_size]]
                batch = rows[batch_starts[:, None] + offsets]
                optimizer.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                optimizer.step()
        model.eval()
        self._model, self._n_channels = model, n_channels

    def score(self, X: np.ndarray, *, context: int) -> np.ndarray:
        """Score every row of ``X`` after its first ``context`` rows.

        Row ``t`` is scored from the window ``X[t - window + 1 : t + 1]``, which the first
        ``context`` rows make available for every scored row.

        Args:
            X: Array of shape ``(n_rows, n_channels)``; its first ``context`` rows are history.
            context: Number of leading history rows; at least ``window - 1``.

        Returns:
            A float64 array of ``len(X) - context`` scores; higher is more anomalous.

        Raises:
            ValueError: ``context`` is below ``window - 1`` or leaves no rows, the channel count
                differs from the fitted one, or ``X`` holds NaN or infinity.
            RuntimeError: The detector is not fitted.
        """
        check_context(self, X, context)
        model = self._model
        if model is None:
            raise RuntimeError(f"{self.name} is not fitted: call fit before score")
        data = self._as_matrix(X)
        if data.shape[1] != self._n_channels:
            raise ValueError(
                f"{self.name} was fitted on {self._n_channels} columns, got {data.shape[1]} columns"
            )
        rows = torch.from_numpy(data)
        last_rows = torch.arange(context, len(data))
        offsets = torch.arange(self.window) - (self.window - 1)
        scores = np.empty(len(last_rows), dtype=np.float64)
        with torch.no_grad():
            for first in range(0, len(last_rows), _SCORE_BATCH):
                ends = last_rows[first : first + _SCORE_BATCH]
                batch = rows[ends[:, None] + offsets]
                reconstructed: torch.Tensor = model(batch)
                error = (reconstructed[:, -1, :] - batch[:, -1, :]) ** 2
                scores[first : first + len(ends)] = error.double().mean(dim=1).numpy()
        return scores

    def state_digest(self) -> str:
        """Return a digest of the kind, the window and every weight of the fitted network.

        Returns:
            A 64-character hex string, equal for two detectors trained identically.

        Raises:
            RuntimeError: The detector is not fitted.
        """
        if self._model is None:
            raise RuntimeError(f"{self.name} is not fitted: call fit before state_digest")
        state = self._model.state_dict()
        return json_digest(
            {
                "kind": self.kind,
                "window": self.window,
                "weights": {
                    key: array_digest(state[key].detach().cpu().numpy()) for key in sorted(state)
                },
            }
        )

    def _build(self, n_channels: int) -> nn.Module:
        """Create the untrained network for ``n_channels`` channels."""
        if self.kind == "cnn":
            return CnnAutoencoder(n_channels, self.size, self.latent)
        return LstmAutoencoder(n_channels, self.size, self.latent)

    def _as_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return a private float32 copy of ``X``, refusing other shapes and non-finite values."""
        data = np.array(X, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(
                f"{self.name} needs a 2-D array of rows and columns, got shape {data.shape}"
            )
        if not np.isfinite(data).all():
            raise ValueError(f"{self.name} needs finite values: X holds NaN or infinity")
        return data
