import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=(2, 2)),
            # nn.BatchNorm2d(32, 0.8, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2)),
            # nn.BatchNorm2d(64, 0.8, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 2048), nn.ReLU(inplace=True), nn.Linear(2048, num_classes))

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv(x)
        # print(x.shape)
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc(x)
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.zeros_(m.bias.data)

