# Source: https://colab.research.google.com/drive/19iOSR-u5s1zoVFdRakg3UtRHJpXuHd0d
# https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_avg_pool2d(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
