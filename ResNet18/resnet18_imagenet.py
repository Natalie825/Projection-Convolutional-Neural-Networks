import math
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import torch.nn.functional as F


class BinConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, binary=False, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BinConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.binary = binary

    def forward(self, input):
        if self.binary:
            input = input.sign()
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ', binary={binary}'
        return s.format(**self.__dict__)


def conv3x3(in_planes, out_planes, stride=1, binary=False):
    """3x3 convolution with padding"""
    return BinConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, binary=binary)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, binary=False):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.conv = conv3x3(inplanes, planes, stride, binary=binary)
        self.prelu = nn.PReLU(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.bn(x)
        out = self.conv(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            shortcut_conv = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, \
                                 stride=1, bias=False)
            downsample = nn.Sequential(
                shortcut_conv,
                nn.BatchNorm2d(planes * block.expansion),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, binary=True))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, binary=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18():
    """
    Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [4, 4, 4, 4])
    return model