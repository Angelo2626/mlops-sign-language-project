"""Defines the SimpleCNN model architecture."""
from torch import nn
from torch.nn import functional as F

class SimpleCNN(nn.Module):
    """Una semplice rete neurale convoluzionale per la classificazione di immagini."""

    def __init__(self, num_classes=25):
        super().__init__()  # Correzione R1725
        # ... (il resto di __init__ rimane uguale)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Defines the forward pass of the model."""  # Correzione C0116
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7 * 7)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
