import torch
import torch.nn as nn


class DeepConvNet(nn.Module):
    def __init__(self, n_ch, n_time, n_classes=2, dropout=0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), padding=(0, 5)),
            nn.Conv2d(25, 25, kernel_size=(n_ch, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 10), padding=(0, 5)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(dropout),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 10), padding=(0, 5)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_time)
            out = self._forward_features(dummy)
            self.flat_dim = out.reshape(1, -1).shape[1]

        self.classifier = nn.Linear(self.flat_dim, n_classes)

    def _forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x
