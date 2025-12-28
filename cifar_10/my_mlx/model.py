import mlx.core as mx
import mlx.nn as nn


# Cifar-10 model definition
class Cifar10MLXClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar10MLXClassifier, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm(num_features=32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm(num_features=64)

        self.fc1 = nn.Linear(input_dims=64 * 8 * 8, output_dims=512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(input_dims=512, output_dims=num_classes)
        self.num_classes = num_classes

    def __call__(self, x):
        x = self.bn1(self.pool1(nn.relu(self.conv1(x))))
        x = self.bn2(self.pool2(nn.relu(self.conv2(x))))
        x = mx.reshape(x, (x.shape[0], -1))  # Flatten
        x = nn.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
