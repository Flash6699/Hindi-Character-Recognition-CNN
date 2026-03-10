import torch
import torch.nn as nn
import torch.nn.functional as F


class HindiCNN(nn.Module):
    """
    CNN for Hindi Character Recognition
    Input: 1x32x32 grayscale image
    """

    def __init__(self, num_classes):
        super(HindiCNN, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # pooling
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))   # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))   # 64x8x8
        x = self.pool(F.relu(self.conv3(x)))   # 128x4x4

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x