import torch
import torch.nn as nn

# -------------------------- 3D Resnet --------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=(1, 1, 1), downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=3, in_channels=3):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(in_channels, self.in_channels, kernel_size=(7, 7, 7), stride=(2, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2, 2, 2))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_cls = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=(1, 1, 1)):
        downsample = None
        if stride != (1, 1, 1) or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion, kernel_size=(1, 1, 1), stride=stride,
                          bias=False),
                nn.BatchNorm3d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_feat = self.avgpool(x)
        x_feat = torch.flatten(x_feat, 1)  # [B, 2048]
        x_cls = self.fc_cls(x_feat)  # [B, 3]
        return x_cls, x_feat


def load_best_model(model_class, model_path, feature_model=None, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature_model is not None:

        model = model_class(feature_model, **kwargs)
    else:

        model = model_class(**kwargs)

    # checkpoint = torch.load(model_path, map_location=device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Loaded best my_model from {model_path}")
    print(f"Model accuracy: {checkpoint['accuracy']:.2f}%")

    return model
def get_resnet3d(args):
    model = ResNet3D(num_classes=3, in_channels=3)
    if args['pretrained']:
        return load_best_model(
                ResNet3D,
                '',
                num_classes=3,
                in_channels=3
            )
    return model
