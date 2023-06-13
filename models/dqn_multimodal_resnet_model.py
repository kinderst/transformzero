import torch
import torch.nn as nn

from models.dqn_resnet_model import ResNet


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        out = x * attention
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attention = self.sigmoid(avg_out + max_out)
        out = x * attention
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)
        self.channel_attention = ChannelAttention(out_channels)

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

        out = self.spatial_attention(out)
        out = self.channel_attention(out)

        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class MultimodalCNN(nn.Module):
    def __init__(self, num_layers, input_shape1, input_shape2, num_actions):
        super(MultimodalCNN, self).__init__()

        self.cnn1 = ResNet(num_layers, input_shape1, num_actions)
        self.cnn2 = ResNet(num_layers, input_shape2, num_actions)

        self.fc = nn.Linear(2 * num_actions, num_actions)

    def forward(self, image1, image2):
        # print("img 1 shape: ", image1.shape)
        # print("img 2 shape: ", image2.shape)
        features1 = self.cnn1(image1)
        features2 = self.cnn2(image2)

        # Concatenate features from both images
        combined_features = torch.cat((features1, features2), dim=1)

        x = self.fc(combined_features)
        return x

# # Usage example
# image1 = torch.randn(1, 3, 32, 32)  # Assuming 32x32 RGB image
# image2 = torch.randn(1, 3, 32, 32)  # Assuming 32x32 RGB image
#
# num_layers = 3
# input_shape1 = (3, 32, 32)
# input_shape2 = (3, 32, 32)
# num_actions = 10
#
# model = MultimodalCNN(num_layers, input_shape1, input_shape2, num_actions)
# output = model(image1, image2)
# print(output.shape)  # Output shape: (1, 10) for 10 classes
