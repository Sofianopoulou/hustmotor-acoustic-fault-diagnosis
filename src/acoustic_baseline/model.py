import torch
import torch.nn as nn


class AcousticCNN1D(nn.Module):
    """Paper-inspired acoustic 1D CNN for six-class motor fault diagnosis.

    Notes
    -----
    The reference paper specifies the filter counts, kernel sizes, stride,
    activation, normalization, and max-pooling configuration, but does not
    explicitly report convolution padding or the exact BatchNorm/ReLU order.
    This implementation uses padding=0 and Conv -> BatchNorm -> ReLU -> Pool.
    """

    def __init__(self, n_classes: int = 6, input_length: int = 1024):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=1, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=16, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            feature_map = self.features(dummy)
            self.feature_shape = tuple(feature_map.shape[1:])
            self.feature_dim = feature_map.flatten(1).shape[1]

        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the flattened acoustic representation before classification."""
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits.

        Softmax is intentionally not applied here because CrossEntropyLoss
        expects raw logits. Apply torch.softmax(logits, dim=1) when explicit
        probabilities are needed.
        """
        return self.classifier(self.extract_features(x))
