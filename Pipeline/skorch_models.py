"""
NOTE:
Changes in this file are not included in grid-search hashes.
`train_select_classifiers.py` currently hashes only `param_grid` contents.
"""

from __future__ import annotations

import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from skorch import NeuralNetBinaryClassifier, NeuralNetRegressor
from skorch.callbacks import Callback, EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DEFAULT_SEED = 42
AHA_TARGET_SCALE = 100.0


def _silent_sink(*args, **kwargs):
    """Pickle-safe no-op callback sink for skorch logging hooks."""
    return None


def set_global_determinism(seed: int = DEFAULT_SEED) -> None:
    """
    Configure Python/NumPy/PyTorch RNGs and deterministic torch backends.
    Call this before each model fit so results do not depend on fit ordering.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Backward compatibility with torch versions that don't expose warn_only.
        torch.use_deterministic_algorithms(True)


class ResetSeedOnTrainBegin(Callback):
    """Reset all RNGs at each fit call for deterministic skorch training."""

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = seed

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        set_global_determinism(self.seed)


def plot_train_valid(history_df, x, train_col, valid_col, ylabel, title, save_path, yscale=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(x, history_df[train_col], label=train_col)
    ax.plot(x, history_df[valid_col], label=valid_col)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_best_estimator_plots(estimator, stats_folder, loss_label="Loss"):
    import pandas as pd

    # Only skorch estimators expose per-epoch training history.
    history = getattr(estimator, "history", None)
    if history is None or not hasattr(history, "to_list"):
        return

    history_df = pd.DataFrame(history.to_list())
    x = history_df["epoch"]
    if {"train_loss", "valid_loss"}.issubset(history_df.columns):
        plot_train_valid(
            history_df,
            x,
            "train_loss",
            "valid_loss",
            loss_label,
            "Loss During Training",
            os.path.join(stats_folder, "best_estimator_loss.png"),
        )
        loss_values = history_df[["train_loss", "valid_loss"]].to_numpy(dtype=float)
        if np.all(loss_values > 0):
            plot_train_valid(
                history_df,
                x,
                "train_loss",
                "valid_loss",
                loss_label,
                "Loss During Training (Log Scale)",
                os.path.join(stats_folder, "best_estimator_loss_log.png"),
                yscale="log",
            )
    if {"train_f1", "valid_f1"}.issubset(history_df.columns):
        plot_train_valid(
            history_df,
            x,
            "train_f1",
            "valid_f1",
            "F1 Macro",
            "F1 Macro During Training",
            os.path.join(stats_folder, "best_estimator_f1.png"),
        )
    if {"train_r2", "valid_r2"}.issubset(history_df.columns):
        plot_train_valid(
            history_df,
            x,
            "train_r2",
            "valid_r2",
            "R2 Score",
            "R2 During Training",
            os.path.join(stats_folder, "best_estimator_r2.png"),
        )


class SequenceStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def _ensure_channel_axis(self, X):
        X = np.asarray(X, dtype=np.float32)
        squeezed = False
        if X.ndim == 2:
            X = X[..., None]
            squeezed = True
        if X.ndim != 3:
            raise ValueError(f"SequenceStandardScaler expects 2D or 3D input, got shape {X.shape}.")
        return X, squeezed

    def fit(self, X, y=None):
        X, _ = self._ensure_channel_axis(X)
        self.scaler.fit(X.reshape(-1, X.shape[-1]))
        return self

    def transform(self, X):
        X, squeezed = self._ensure_channel_axis(X)
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape).astype(np.float32)
        if squeezed:
            X_scaled = X_scaled[..., 0]
        return X_scaled


def _pool_sequence_hidden_states(out, readout_mode: str, attention: nn.Module | None = None):
    if readout_mode == "last":
        return out[:, -1]
    if readout_mode == "mean":
        return out.mean(dim=1)
    if readout_mode == "max":
        return out.max(dim=1).values
    if readout_mode == "attention":
        if attention is None:
            raise ValueError("attention module is required when readout_mode='attention'.")
        scores = attention(out).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), out).squeeze(1)
    raise ValueError(f"Unsupported readout_mode: {readout_mode}")


def make_regressor_net(module, module_kwargs=None, device=None):
    module_kwargs = {} if module_kwargs is None else dict(module_kwargs)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return NeuralNetRegressor(
        module=module,
        callbacks=[
            ("deterministic_seed", ResetSeedOnTrainBegin(seed=DEFAULT_SEED)),
            (
                "train_r2",
                EpochScoring(
                    scoring="r2",
                    on_train=True,
                    lower_is_better=False,
                    name="train_r2",
                    use_caching=False,
                ),
            ),
            (
                "valid_r2",
                EpochScoring(
                    scoring="r2",
                    on_train=False,
                    lower_is_better=False,
                    name="valid_r2",
                    use_caching=False,
                ),
            ),
            (
                "early_stopping",
                EarlyStopping(
                    monitor="valid_loss",
                    threshold=1e-4,
                    threshold_mode="rel",
                    lower_is_better=True,
                    sink=_silent_sink,
                    load_best=True,
                ),
            ),
        ],
        criterion=nn.MSELoss,
        optimizer=torch.optim.AdamW,
        iterator_train__shuffle=True,
        train_split=ValidSplit(0.2, random_state=42),
        device=device,
        verbose=0,
        callbacks__print_log=None,
        **{f"module__{k}": v for k, v in module_kwargs.items()},
    )


def make_bce_net(module):
    if module is XGBoostSequenceClassifier:
        return XGBoostSequenceClassifier()

    # Build a fresh skorch estimator each time to avoid shared mutable state across trials.
    # Disable skorch's default PrintLog callback so estimator state does not capture
    # a possibly patched `print` callable from Ray and remains quiet/picklable.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = NeuralNetBinaryClassifier(
        module=module,
        callbacks=[
            ("deterministic_seed", ResetSeedOnTrainBegin(seed=DEFAULT_SEED)),
            (
                "train_f1",
                EpochScoring(
                    scoring="f1_macro",
                    on_train=True,
                    lower_is_better=False,
                    name="train_f1",
                ),
            ),
            (
                "valid_f1",
                EpochScoring(
                    scoring="f1_macro",
                    on_train=False,
                    lower_is_better=False,
                    name="valid_f1",
                ),
            ),
            (
                "early_stopping",
                EarlyStopping(
                    monitor="valid_loss",
                    threshold=1e-4,
                    threshold_mode="rel",
                    lower_is_better=True,
                    sink=_silent_sink,
                    load_best=True,
                ),
            )
        ],
        criterion=nn.BCEWithLogitsLoss,
        optimizer=torch.optim.AdamW,
        iterator_train__shuffle=True,
        train_split=ValidSplit(0.2, stratified=True, random_state=42),
        device=device,
        verbose=0,
        callbacks__print_log=None,
    )
    net.threshold = 0.5
    return net


class LSTMSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        readout_mode: str = "last",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.readout_mode = readout_mode

        self.lstm = None
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * direction_factor, 1)
        self.attention = nn.Linear(hidden_size * direction_factor, 1) if self.readout_mode == "attention" else None

    def _build_lstm(self, input_size, device):
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        ).to(device)

    def forward(self, x):
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        if self.lstm is None:
            self._build_lstm(x.shape[-1], x.device)

        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        pooled = _pool_sequence_hidden_states(out, self.readout_mode, self.attention)
        return self.classifier(pooled).squeeze(-1)


class RNNSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        nonlinearity: str = "tanh",
        readout_mode: str = "last",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity
        self.readout_mode = readout_mode

        self.rnn = None
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * direction_factor, 1)
        self.attention = nn.Linear(hidden_size * direction_factor, 1) if self.readout_mode == "attention" else None

    def _build_rnn(self, input_size, device):
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            nonlinearity=self.nonlinearity,
            batch_first=True,
        ).to(device)

    def forward(self, x):
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        if self.rnn is None:
            self._build_rnn(x.shape[-1], x.device)

        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        pooled = _pool_sequence_hidden_states(out, self.readout_mode, self.attention)
        return self.classifier(pooled).squeeze(-1)


class MLPSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        layers = [nn.LazyLinear(hidden_size), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.float()
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        hidden = self.mlp(x)
        return self.classifier(hidden).squeeze(-1)



class GRUSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        readout_mode: str = "last",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.readout_mode = readout_mode

        self.gru = None
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * direction_factor, 1)
        self.attention = nn.Linear(hidden_size * direction_factor, 1) if self.readout_mode == "attention" else None

    def _build_gru(self, input_size, device):
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        ).to(device)

    def forward(self, x):
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        if self.gru is None:
            self._build_gru(x.shape[-1], x.device)

        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        pooled = _pool_sequence_hidden_states(out, self.readout_mode, self.attention)
        return self.classifier(pooled).squeeze(-1)


class GRUSequenceRegressor(nn.Module):
    def __init__(
        self,
        *,
        input_size: int | None = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        return_last_step: bool = False,
        readout_mode: str = "last",
        keep_output_dim: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.return_last_step = return_last_step
        self.readout_mode = readout_mode
        self.keep_output_dim = keep_output_dim

        self.input_size = input_size
        self.gru = None
        self.skip = None
        self.regressor = nn.Linear(hidden_size, 2)
        self.attention = nn.Linear(hidden_size, 1) if self.readout_mode == "attention" else None

        if self.input_size is not None:
            self._build_layers(self.input_size, torch.device("cpu"))

    def _build_layers(self, input_size, device):
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
        ).to(device)
        self.skip = nn.Linear(input_size, 2).to(device)

    def _decode_soft_aha_from_severity(self, raw_outputs):
        healthy_probability = torch.sigmoid(raw_outputs[..., 0])
        affected_severity = torch.sigmoid(raw_outputs[..., 1]) * AHA_TARGET_SCALE
        severity = (1.0 - healthy_probability) * affected_severity
        aha = AHA_TARGET_SCALE - severity
        if self.keep_output_dim:
            return aha.unsqueeze(-1)
        return aha

    def _pool_hidden_states(self, out, skip_out):
        if self.readout_mode == "last":
            return out[:, -1], skip_out[:, -1]
        if self.readout_mode == "mean":
            return out.mean(dim=1), skip_out.mean(dim=1)
        if self.readout_mode == "max":
            pooled_out = out.max(dim=1).values
            pooled_skip = skip_out.max(dim=1).values
            return pooled_out, pooled_skip
        if self.readout_mode == "attention":
            scores = self.attention(out).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            pooled_out = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
            pooled_skip = torch.bmm(weights.unsqueeze(1), skip_out).squeeze(1)
            return pooled_out, pooled_skip
        raise ValueError(f"Unsupported readout_mode: {self.readout_mode}")

    def _causal_pooled_hidden_states(self, out, skip_out):
        # Prefix-causal pooling: prediction at time t uses only steps <= t.
        if self.readout_mode == "last":
            return out, skip_out
        if self.readout_mode == "mean":
            denom = torch.arange(1, out.shape[1] + 1, device=out.device, dtype=out.dtype).view(1, -1, 1)
            pooled_out = out.cumsum(dim=1) / denom
            pooled_skip = skip_out.cumsum(dim=1) / denom
            return pooled_out, pooled_skip
        if self.readout_mode == "max":
            return out.cummax(dim=1).values, skip_out.cummax(dim=1).values
        if self.readout_mode == "attention":
            scores = self.attention(out).squeeze(-1)
            t_steps = out.shape[1]
            causal_mask = torch.tril(torch.ones(t_steps, t_steps, device=out.device, dtype=torch.bool))
            score_matrix = scores.unsqueeze(1).expand(-1, t_steps, -1)
            score_matrix = score_matrix.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
            weights = torch.softmax(score_matrix, dim=-1)
            pooled_out = torch.bmm(weights, out)
            pooled_skip = torch.bmm(weights, skip_out)
            return pooled_out, pooled_skip
        raise ValueError(f"Unsupported readout_mode: {self.readout_mode}")

    def _sequence_predictions(self, out, skip_out):
        pooled_out, pooled_skip = self._causal_pooled_hidden_states(out, skip_out)
        return self._decode_soft_aha_from_severity(self.regressor(pooled_out) + pooled_skip)

    def forward(self, x):
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        if self.gru is None or self.skip is None:
            self._build_layers(x.shape[-1], x.device)

        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        skip_out = self.skip(x)
        if self.return_last_step:
            pooled_out, pooled_skip = self._pool_hidden_states(out, skip_out)
            return self._decode_soft_aha_from_severity(self.regressor(pooled_out) + pooled_skip)
        return self._sequence_predictions(out, skip_out)


def _flatten_windows(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    return X


class XGBoostSequenceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_ = None

    def _make_model(self):
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, X, y):
        X = _flatten_windows(X)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.model_ = self._make_model()
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        X = _flatten_windows(X)
        return self.model_.predict_proba(X)

    def predict(self, X):
        X = _flatten_windows(X)
        return self.model_.predict(X)


class Conv1DSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        channels: int = 32,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        self.features = None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * 2, 1)

    def _build_features(self, in_channels):
        padding = self.kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, self.channels, self.kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(self.channels, self.channels * 2, self.kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(1)
        else:
            x = x.permute(0, 2, 1)

        if self.features is None:
            self._build_features(x.shape[1])
            self.features = self.features.to(x.device)

        feat = self.features(x).squeeze(-1)
        return self.classifier(self.dropout(feat)).squeeze(-1)




def _sinusoidal_positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        patch_size: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        self.embed = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # Pre-norm encoder (`norm_first=True`) is incompatible with nested tensor
        # acceleration; disable it explicitly to avoid repeated runtime warnings.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers,
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

    def _build_embed(self, input_size, device):
        self.embed = nn.Conv1d(
            input_size,
            self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).to(device)

    def forward(self, x):
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        x = x.permute(0, 2, 1)

        if self.embed is None:
            self._build_embed(x.shape[1], x.device)

        tokens = self.embed(x).permute(0, 2, 1)
        pos = _sinusoidal_positional_encoding(tokens.shape[1], tokens.shape[2], tokens.device)
        encoded = self.encoder(tokens + pos.unsqueeze(0))
        pooled = self.dropout(encoded.mean(dim=1))
        return self.classifier(pooled).squeeze(-1)




def _estimate_spectral_norm(W: torch.Tensor, n_iter: int = 15) -> torch.Tensor:
    v = torch.randn(W.shape[1], device=W.device)
    v = v / (v.norm() + 1e-12)
    for _ in range(n_iter):
        v = W.T @ (W @ v)
        v = v / (v.norm() + 1e-12)
    return (W @ v).norm()


class ReservoirSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        reservoir_size: int = 200,
        spectral_radius: float = 0.9,
        leak_rate: float = 1.0,
        input_scaling: float = 0.5,
        downsample: int = 16,
    ):
        super().__init__()

        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.downsample = downsample

        self.W_in = None
        self.W = None
        self.bias = None

        self.classifier = nn.Linear(self.reservoir_size, 1)

    def _build_reservoir(self, input_size, device):
        W_in = (torch.rand(input_size, self.reservoir_size, device=device) * 2 - 1) * self.input_scaling

        W = torch.randn(self.reservoir_size, self.reservoir_size, device=device)
        with torch.no_grad():
            sn = _estimate_spectral_norm(W)
            W *= self.spectral_radius / (sn + 1e-12)

        self.W_in = W_in
        self.W = W
        self.bias = torch.zeros(self.reservoir_size, device=device)

    def forward(self, x):
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        if self.downsample > 1:
            x = F.avg_pool1d(x.permute(0, 2, 1), self.downsample).permute(0, 2, 1)

        if self.W_in is None:
            self._build_reservoir(x.shape[-1], x.device)

        state = x.new_zeros(x.shape[0], self.reservoir_size)
        for t in range(x.shape[1]):
            u = x[:, t, :]
            pre = u @ self.W_in + state @ self.W + self.bias
            state = (1 - self.leak_rate) * state + self.leak_rate * torch.tanh(pre)

        return self.classifier(state).squeeze(-1)
