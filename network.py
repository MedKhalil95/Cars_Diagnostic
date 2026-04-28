import torch
import torch.nn as nn


class CarDiagnosticNet(nn.Module):
    """
    Feed-forward network for car engine-health classification.

    Improvements over v1:
    - BatchNorm after each linear layer → faster, stabler convergence
    - Kaiming weight initialisation  → better gradient flow from step 0
    - Configurable num_classes        → no magic numbers
    - Deeper hidden path (128 → 64 → 32) for richer feature extraction
    """

    def __init__(self, input_size: int, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Linear(32, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)