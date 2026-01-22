from __future__ import annotations

from torch import nn


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
