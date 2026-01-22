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
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.classifier(last)
