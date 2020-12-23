from lib import cnn_bn_relu, StatsPooling, AMLinear, Fbank, SpectrumAug
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, activation='relu'):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(4*growth_rate)
        self.conv2 = nn.Conv1d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'prelu':
            self.activation = torch.nn.PReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, activation='relu'):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm1d(in_planes)
        # self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=2, stride=2, bias=False)

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'prelu':
            self.activation = torch.nn.PReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.conv(self.activation(self.bn(x)))
        # out = F.avg_pool1d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block,
                 num_blocks,
                 n_classes,
                 fbank_config=None,
                 specaug_config=None,
                 activation='relu',
                 loss='amsoftmax',
                 m=0.35,
                 use_wav=False,
                 two_layer_fc=False,
                 mfcc_dim=41, embedding_size=256,
                 growth_rate=40, reduction=0.5):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        trans = []
        if fbank_config:
            trans.append(Fbank(**fbank_config))
            mfcc_dim = fbank_config['n_mels']
        if specaug_config:
            trans.append(SpectrumAug(**specaug_config))
        if trans:
            self.trans = nn.Sequential(*trans)
        else:
            self.trans = None

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv1d(mfcc_dim, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, num_blocks[0], activation=activation)
        num_planes += num_blocks[0] * growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes, activation=activation)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, num_blocks[1], activation=activation)
        num_planes += num_blocks[1] * growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes, activation=activation)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, num_blocks[2], activation=activation)
        num_planes += num_blocks[2] * growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes, activation=activation)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, num_blocks[3], activation=activation)
        num_planes += num_blocks[3] * growth_rate

        self.bn = nn.BatchNorm1d(num_planes)


        # self.linear = nn.Linear(num_planes, num_classes)

        if not two_layer_fc:
            self.fc = nn.Linear(num_planes*2, embedding_size)
        else:
            self.fc = nn.Sequential(
                nn.Linear(num_planes*2, 512),
                nn.BatchNorm1d(512),
                torch.nn.ReLU(),
                nn.Linear(512, embedding_size)
            )
        self.stats_pooling = StatsPooling()
        if loss == 'amsoftmax':
            print('using amsoftmax')
            self.cls_layer = AMLinear(embedding_size, n_classes, m=m)
        elif loss == 'softmax':
            self.cls_layer = nn.Linear(embedding_size, n_classes)
        else:
            raise NotImplementedError

    def _make_dense_layers(self, block, in_planes, nblock, activation):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, activation=activation))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        if self.trans:
            x = self.trans(x)
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = F.relu(self.bn(self.dense4(out)))

        out = self.stats_pooling(out)
        out = self.fc(out)
        if y is not None:
            out = self.cls_layer(out, y)
        else:
            out = self.cls_layer(out)
        # out = self.cls_layer(out, y)
        return out

    def extract(self, x):
        if self.trans:
            x = self.trans(x)
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = F.relu(self.bn(self.dense4(out)))

        out = self.stats_pooling(out)
        out = self.fc(out)
        return out


def DenseNet121(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 24, 16], **kwargs)

def DenseNet169(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 32, 32], **kwargs)


def DenseNet201(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 48, 32], **kwargs)

def DenseNet161(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 36, 24], **kwargs)
#
def DenseNet80(**kwargs):
    return DenseNet(Bottleneck, [6, 10, 14, 10], **kwargs)



class XvectorSeq(nn.Module):
    def __init__(self, n_classes, margin=0.35, embedding_layer=None):
        super().__init__()
        self.frame_layers = nn.Sequential(
            # torchaudio.transforms.MFCC(log_mels=True, melkwargs={'hop_length': 160, 'n_mels': 40},),
            # CMVN(),
            Fbank(80),
            SpectrumAug(max_freq_mask_len=27, max_time_mask_len=100, n_freq_mask=1),
            cnn_bn_relu(80, 512, 5, dilation=1),
            cnn_bn_relu(512, 512, 3, dilation=2),
            cnn_bn_relu(512, 512, 3, dilation=3),
            cnn_bn_relu(512, 512, 1, dilation=1),
            cnn_bn_relu(512, 1500, 1, dilation=1),
        )
        self.utt_layers = nn.Sequential(
            nn.Linear(3000, 512),
            nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            nn.Linear(512, 128),
        )

        self.am_linear = AMLinear(128, n_classes, m=margin)
        self.stats_pooling = StatsPooling()
        self.embedding_layer = embedding_layer

    def forward(self, x, y):
        x = self.frame_layers(x)
        x = self.stats_pooling(x)
        embedding = self.utt_layers(x)
        return self.am_linear(embedding, y)

    def extract(self, x):
        x = self.frame_layers(x)
        x = self.stats_pooling(x)
        embedding = self.utt_layers[:self.embedding_layer](x)
        return embedding


def choose_model(model_config, embedding_layer=None):
    print(model_config['model'])
    if model_config['model'] == 'Xvector':
        print('using xvector')
        model_config.pop('model')
        model = XvectorSeq(**model_config, embedding_layer=embedding_layer)
    elif model_config['model'] == 'densenet121_1d':
        print('using densenet121_1d')
        model_config.pop('model')
        model = DenseNet(Bottleneck, [6, 12, 24, 16], **model_config)
    else:
        print(model_config['model'])
        raise NotImplementedError
    return model