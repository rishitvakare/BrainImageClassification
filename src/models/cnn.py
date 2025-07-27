import torch.nn as nn

class CNN(nn.Module):
  def __init__(self, num_classes=4):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),

      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),

      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
    )

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Dropout(0.5),
      nn.Linear(128 * 28 * 28, 256),
      nn.ReLU(inplace=True),
      nn.Dropout(0.5),
      nn.Linear(256, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x
