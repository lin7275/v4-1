import torch
import torch.nn as nn
# from torchvision.models.etc import load_state_dict_from_url
from lib import StatsPooling
from lib import AMLinear
from lib import SpectrumAug, Fbank


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class SplitConv(nn.Module):
    def __init__(self, n_splits, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(**kwargs) for _ in range(n_splits)])
        self.n_splits = n_splits

    def forward(self, x):
        # X 64X40X400
        out = []
        split_size = x.shape[2] // self.n_splits
        for x_split, conv in zip(x.split(split_size, dim=2), self.convs):
            # print(x_split.shape)
            out.append(conv(x_split))
            # print(out[-1].shape)
        return torch.cat(out, dim=1)


def split_conv3x3(n_splits, in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return SplitConv(n_splits, in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def split_conv1x1(n_splits, in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return SplitConv(n_splits, in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    # expansion = 2
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SimpleSplit(nn.Module):
    def __init__(self, n_split):
        super().__init__()
        self.n_split = n_split

    def forward(self, x):
        # add overlap
        # print(x.shape)
        # x1 = x[:, :, :(x.shape[2] // self.n_split + 3), :]
        # x2 = x[:, :, (x.shape[2] // self.n_split - 3):, :]
        # if x.shape[2] == 41:
        #     x = x[:, :, 1:, :]
        return x.reshape(x.shape[0], -1, x.shape[2] // self.n_split, x.shape[3])

    def __str__(self):
        return f'n_split {self.n_split}'
# x = np.arange(8)
# x1 = x[:(x.shape[0] // 2 + 2)]
# x2 = x[(x.shape[0] // 2 - 2):]
class ResNet(nn.Module):
    def __init__(self, block, layers, n_classes,
                 n_split=2,
                 fbank_config=None,
                 specaug_config=None,
                 m=0.35, w_base=32, mfcc_dim=41,
                 pooling_layer=None,
                 pooling_config=None,
                 embedding_size=256,
                 loss='amsoftmax',
                 zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 use_wav=False,
                 norm_layer=None):
        super(ResNet, self).__init__()
        trans = []
        if fbank_config:
            trans.append(Fbank(**fbank_config))
            mfcc_dim = fbank_config['n_mels']
        if specaug_config:
            print('using specaug')
            trans.append(SpectrumAug(**specaug_config))
        if trans:
            self.trans = nn.Sequential(*trans)
        else:
            self.trans = None

        self.loss = loss
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = w_base
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if n_split != 1:
            # self.conv1 = split_conv3x3(n_split, 1, self.inplanes)
            self.conv1 = nn.Sequential(
                SimpleSplit(n_split),
                nn.Conv2d(n_split, self.inplanes  * n_split, kernel_size=3,
                                   bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3,
                                   bias=False)
        self.inplanes = self.inplanes * n_split
        # self.inplanes = self.inplanes
        self.bn1 = norm_layer(self.inplanes)

        self.relu = nn.ReLU(inplace=True)


        self.layer1 = self._make_layer(block, w_base, layers[0])
        self.layer2 = self._make_layer(block, 2*w_base, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        # self.layer2 = self._make_layer(block, 2*w_base, layers[1], stride=1,
        #                                dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 4*w_base, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 8*w_base, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        if not pooling_layer:
            self.stats_pooling = StatsPooling()
        else:
            raise ValueError(f"{pooling_config['pooling_layer']} not supported")

        # self.fc = nn.Linear(mfcc_dim//5*w_base*10*block.expansion, embedding_size)
        self.fc = nn.Linear(6144, embedding_size)
        # self.fc = nn.Linear(1536, embedding_size)

        if self.loss == 'amsoftmax':
            print('using amsoftmax')
            self.cls_layer = AMLinear(embedding_size, n_classes, m=m)
        elif loss == 'softmax':
            self.cls_layer = nn.Linear(embedding_size, n_classes)
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, x, y=None):
        x = x[:, None, ...]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.stats_pooling(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        if y is not None:
            x = self.cls_layer(x, y)
        else:
            x = self.cls_layer(x)
        return x

    def forward(self, x, y=None):
        if self.trans:
            x = self.trans(x)
        x = x[:, None, ...]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.stats_pooling(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        if y is not None:
            x = self.cls_layer(x, y)
        else:
            x = self.cls_layer(x)
        return x

    def extract(self, x):
        if self.trans:
            x = self.trans(x)
        x = x[:, None, ...]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.stats_pooling(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    # Allow for accessing forward method in a inherited class
    # forward = _forward


class ResNetDePloy(ResNet):

    def forward(self, x):
        x = x.transpose(1, 0)[None, :, :]
        x = x[:, None, ...]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.stats_pooling(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _resnet_deploy(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetDePloy(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet101deploy(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_deploy('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


# def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-50 32x4d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
#                    pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 16
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 3, 4, 3],
                   pretrained, progress, w_base=16, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


if __name__ == '__main__':
    m = resnet18(n_classes=1000, m=0.35)
    x = torch.randn(2, 1, 40, 400)
    # print(torch.randn(2).get_device())
    print(m)