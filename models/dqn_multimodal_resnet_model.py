import torch
import torch.nn as nn

from models.dqn_resnet_model import ResNet
from models.min_gpt import NewGELU


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
        # self.relu = nn.ReLU(inplace=True)
        self.relu = NewGELU()
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = NewGELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.spatial_attention = SpatialAttention(out_channels)
        # self.channel_attention = ChannelAttention(out_channels)

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

        # out = self.spatial_attention(out)
        # out = self.channel_attention(out)

        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class MultimodalCNN(nn.Module):
    def __init__(self, num_layers, input_shapes, num_actions, dropout_rate=0.1):
        super(MultimodalCNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modalities = list(input_shapes.keys())
        self.cnns = nn.ModuleDict({modality: ResNet(num_layers, input_shapes[modality], num_actions) for modality in self.modalities})
        self.layer_norm1 = nn.LayerNorm(len(self.modalities) * num_actions)
        # ((len(self.modalities) * num_actions) + num_actions) // 2 attempt to step neurons down proportionally
        num_between = ((len(self.modalities) * num_actions) + num_actions) // 2
        self.fc1 = nn.Linear(len(self.modalities) * num_actions, num_between)
        self.layer_norm2 = nn.LayerNorm(num_between)
        # self.relu = nn.ReLU()
        self.gelu = NewGELU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(num_between, num_actions)

    def forward(self, state_batch):
        tensor_batch = {modality: torch.cat([torch.tensor(sample[modality], device=self.device, dtype=torch.float32).unsqueeze(0)
                                             for sample in state_batch]) for modality in self.modalities}

        features = []
        for modality in self.modalities:
            cnn_output = self.cnns[modality](tensor_batch[modality])
            features.append(cnn_output)

        combined_features = torch.cat(features, dim=1)
        normalized_features1 = self.layer_norm1(combined_features)
        x = self.fc1(normalized_features1)
        normalized_features2 = self.layer_norm2(x)
        # x = self.relu(normalized_features2)
        x = self.gelu(normalized_features2)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FCModule(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FCModule, self).__init__()
        layers = []
        prev_size = input_size[0]  # because passing a tuple i.e. (25,)
        for size in hidden_sizes:
            size = size[0]  # because it some reason converts to [np.array([14], dtype=int32)] ?
            layers.append(nn.LayerNorm(prev_size))
            layers.append(nn.Linear(prev_size, size))
            layers.append(NewGELU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MultimodalResnetAndFC(nn.Module):
    def __init__(self, num_layers, input_shapes, num_actions, dropout_rate=0.1):
        super(MultimodalResnetAndFC, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.img_modalities = [modality for modality in input_shapes.keys() if modality.startswith("img")]
        print("num img modalities: ", len(self.img_modalities))
        self.flat_modalities = [modality for modality in input_shapes.keys() if modality.startswith("flat")]
        print("num flat modalities: ", len(self.flat_modalities))
        # Image modalities
        self.img_cnns = nn.ModuleDict({
            modality: ResNet(num_layers, input_shapes[modality], num_actions)
            for modality in self.img_modalities
        })

        # Flat data modalities
        self.flat_fc_layers = nn.ModuleDict({
            modality: FCModule(
                input_shapes[modality],
                [(input_shapes[modality] + num_actions) // 2],
                num_actions
            )
            for modality in self.flat_modalities
        })

        # Fusion layers
        fusion_input_size = (len(self.img_modalities) + len(self.flat_modalities)) * num_actions
        self.layer_norm1 = nn.LayerNorm(fusion_input_size)
        self.gelu = NewGELU()
        self.fc1 = nn.Linear(fusion_input_size, num_actions)
        # num_between = (fusion_input_size + num_actions) // 2
        # self.fc1 = nn.Linear(fusion_input_size, num_between)
        # self.layer_norm2 = nn.LayerNorm(num_between)
        # self.gelu = NewGELU()
        # self.dropout = nn.Dropout(p=dropout_rate)
        # self.fc2 = nn.Linear(num_between, num_actions)
        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, state_batch):
        img_tensor_batch = {
            modality: torch.cat([
                torch.tensor(sample[modality], device=self.device, dtype=torch.float32).unsqueeze(0)
                for sample in state_batch
            ])
            for modality in self.img_modalities
        }

        flat_tensor_batch = {
            modality: torch.cat([
                torch.tensor(sample[modality], device=self.device, dtype=torch.float32).unsqueeze(0)
                for sample in state_batch
            ])
            for modality in self.flat_modalities
        }

        img_features = []
        for modality in self.img_modalities:
            cnn_output = self.img_cnns[modality](img_tensor_batch[modality])
            img_features.append(cnn_output)

        flat_features = []
        for modality in self.flat_modalities:
            fc_output = self.flat_fc_layers[modality](flat_tensor_batch[modality])
            flat_features.append(fc_output)

        combined_features = torch.cat(img_features + flat_features, dim=1)
        normalized_features1 = self.layer_norm1(combined_features)
        x = self.gelu(normalized_features1)
        x = self.fc1(x)
        # normalized_features2 = self.layer_norm2(x)
        # x = self.gelu(normalized_features2)
        # x = self.dropout(x)
        # x = self.fc2(x)
        return x
