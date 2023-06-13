import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_layers, input_shape, num_actions):
        super(ResNet, self).__init__()
        self.num_layers = num_layers
        self.input_shape = input_shape

        self.conv = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.residual_layers = self.create_residual_layers(64)

        self.fc = nn.Linear(64 * input_shape[1] * input_shape[2], num_actions)

    def create_residual_layers(self, channels):
        layers = []
        for _ in range(self.num_layers):
            layers.append(ResidualBlock(channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.residual_layers(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Example usage
# num_layers = 3  # Specify the number of layers in the ResNet
# input_shape = (3, 25, 25)
# num_actions = 4  # Specify the number of output actions
#
# model = ResNet(num_layers, input_shape, num_actions)
# print(model)
