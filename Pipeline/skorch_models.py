from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LSTMSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = None
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * direction_factor, 1)

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

        out, _ = self.lstm(x)
        return self.classifier(out[:, -1]).squeeze(-1)



class GRUSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.gru = None
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * direction_factor, 1)

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

        out, _ = self.gru(x)
        return self.classifier(out[:, -1]).squeeze(-1)




class RNNSequenceClassifier(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        nonlinearity: str = "tanh",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity

        self.rnn = None
        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * direction_factor, 1)

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

        out, _ = self.rnn(x)
        return self.classifier(out[:, -1]).squeeze(-1)




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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
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

