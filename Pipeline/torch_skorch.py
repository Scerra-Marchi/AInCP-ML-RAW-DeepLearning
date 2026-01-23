from __future__ import annotations

from torch import nn
import torch
import torch.nn.functional as F


class LSTMSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * direction_factor, num_classes)

    def forward(self, x):  # skorch passes numpy -> torch.Tensor
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.classifier(last)


class GRUSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * direction_factor, num_classes)

    def forward(self, x):
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.classifier(last)


class RNNSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        nonlinearity: str = "tanh",
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * direction_factor, num_classes)

    def forward(self, x):
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.classifier(last)


class Conv1DSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        channels: int = 32,
        kernel_size: int = 7,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(channels, channels * 2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * 2, num_classes)

    def forward(self, x):
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, channels=1, seq_len)
        elif x.ndim == 3:
            x = x.permute(0, 2, 1)  # (batch, channels=n_feat, seq_len)
        feat = self.features(x).squeeze(-1)
        feat = self.dropout(feat)
        return self.classifier(feat)


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
        input_size: int = 1,
        num_classes: int = 2,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        patch_size: int = 32,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed = nn.Conv1d(
            in_channels=input_size,
            out_channels=d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        tokens = self.embed(x)  # (batch, d_model, n_tokens)
        tokens = tokens.permute(0, 2, 1)  # (batch, n_tokens, d_model)
        pos = _sinusoidal_positional_encoding(tokens.shape[1], tokens.shape[2], tokens.device)
        tokens = tokens + pos.unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


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
        input_size: int = 1,
        num_classes: int = 2,
        reservoir_size: int = 200,
        spectral_radius: float = 0.9,
        leak_rate: float = 1.0,
        input_scaling: float = 0.5,
        downsample: int = 16,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.reservoir_size = int(reservoir_size)
        self.leak_rate = float(leak_rate)
        self.downsample = int(downsample)

        W_in = (torch.rand(self.input_size, self.reservoir_size) * 2 - 1) * float(input_scaling)
        W = torch.randn(self.reservoir_size, self.reservoir_size)
        with torch.no_grad():
            sn = _estimate_spectral_norm(W)
            W = W * (float(spectral_radius) / (sn + 1e-12))

        self.register_buffer("W_in", W_in)
        self.register_buffer("W", W)
        self.register_buffer("bias", torch.zeros(self.reservoir_size))
        self.classifier = nn.Linear(self.reservoir_size, num_classes)

    def forward(self, x):
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        if self.downsample > 1:
            x_pool = x.permute(0, 2, 1)  # (batch, channels, seq_len)
            x_pool = F.avg_pool1d(x_pool, kernel_size=self.downsample, stride=self.downsample)
            x = x_pool.permute(0, 2, 1)  # (batch, new_len, channels)

        state = x.new_zeros(x.shape[0], self.reservoir_size)
        for t in range(x.shape[1]):
            u = x[:, t, :]  # (batch, input_size)
            pre = u @ self.W_in + state @ self.W + self.bias
            new_state = torch.tanh(pre)
            state = (1.0 - self.leak_rate) * state + self.leak_rate * new_state
        return self.classifier(state)
