import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet-style compact CNN.

    Input:
        batch x 1 x channels x time
    """

    def __init__(
        self,
        n_ch,
        n_time,
        n_classes=2,
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        dropout=0.5,
    ):
        super().__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(
                1,
                F1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                F1,
                F1 * D,
                kernel_size=(n_ch, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(
                F1 * D,
                F1 * D,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F1 * D,
                bias=False,
            ),
            nn.Conv2d(
                F1 * D,
                F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_time)
            out = self._forward_features(dummy)
            self.flat_dim = out.reshape(1, -1).shape[1]

        self.classifier = nn.Linear(self.flat_dim, n_classes)

    def _forward_features(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x
