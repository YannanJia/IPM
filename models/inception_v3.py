import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.size()
    if shape[2] is None or shape[3] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[2], kernel_size[0]),
                           min(shape[3], kernel_size[1])]
    return kernel_size_out

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionV3Base(nn.Module):
    def __init__(self, min_depth=16, depth_multiplier=1.0):
        super(InceptionV3Base, self).__init__()
        self.min_depth = min_depth
        self.depth_multiplier = depth_multiplier
        self.depth = lambda d: max(int(d * depth_multiplier), min_depth)
        
        self.Conv2d_1a_3x3 = BasicConv2d(3, self.depth(32), 3, stride=2, padding=0)
        self.Conv2d_2a_3x3 = BasicConv2d(self.depth(32), self.depth(32), 3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(self.depth(32), self.depth(64), 3, stride=1, padding=1)
        self.MaxPool_3a_3x3 = nn.MaxPool2d(3, stride=2, padding=0)
        self.Conv2d_3b_1x1 = BasicConv2d(self.depth(64), self.depth(80), 1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(self.depth(80), self.depth(192), 3, stride=1, padding=0)
        self.MaxPool_5a_3x3 = nn.MaxPool2d(3, stride=2, padding=0)

        self.Mixed_5b = self._make_Mixed_5b()
        self.Mixed_5c = self._make_Mixed_5c()
        self.Mixed_5d = self._make_Mixed_5d()
        self.Mixed_6a = self._make_Mixed_6a()
        self.Mixed_6b = self._make_Mixed_6b()
        self.Mixed_6c = self._make_Mixed_6c()
        self.Mixed_6d = self._make_Mixed_6d()
        self.Mixed_6e = self._make_Mixed_6e()
        self.Mixed_7a = self._make_Mixed_7a()
        self.Mixed_7b = self._make_Mixed_7b()
        self.Mixed_7c = self._make_Mixed_7c()

    def _make_Mixed_5b(self):
        layers = nn.ModuleDict()
        layers['Branch_0'] = BasicConv2d(self.depth(192), self.depth(64), 1)
        layers['Branch_1'] = nn.Sequential(
            BasicConv2d(self.depth(192), self.depth(48), 1),
            BasicConv2d(self.depth(48), self.depth(64), 5, padding=2)
        )
        layers['Branch_2'] = nn.Sequential(
            BasicConv2d(self.depth(192), self.depth(64), 1),
            BasicConv2d(self.depth(64), self.depth(96), 3, padding=1),
            BasicConv2d(self.depth(96), self.depth(96), 3, padding=1)
        )
        layers['Branch_3'] = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(self.depth(192), self.depth(32), 1)
        )
        return layers

    def _make_Mixed_5c(self):
        layers = nn.ModuleDict()
        layers['Branch_0'] = BasicConv2d(self.depth(256), self.depth(64), 1)
        layers['Branch_1'] = nn.Sequential(
            BasicConv2d(self.depth(256), self.depth(48), 1),
            BasicConv2d(self.depth(48), self.depth(64), 5, padding=2)
        )
        layers['Branch_2'] = nn.Sequential(
            BasicConv2d(self.depth(256), self.depth(64), 1),
            BasicConv2d(self.depth(64), self.depth(96), 3, padding=1),
            BasicConv2d(self.depth(96), self.depth(96), 3, padding=1)
        )
        layers['Branch_3'] = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(self.depth(256), self.depth(64), 1)
        )
        return layers

    def _make_Mixed_5d(self):
        layers = nn.ModuleDict()
        layers['Branch_0'] = BasicConv2d(self.depth(288), self.depth(64), 1)
        layers['Branch_1'] = nn.Sequential(
            BasicConv2d(self.depth(288), self.depth(48), 1),
            BasicConv2d(self.depth(48), self.depth(64), 5, padding=2)
        )
        layers['Branch_2'] = nn.Sequential(
            BasicConv2d(self.depth(288), self.depth(64), 1),
            BasicConv2d(self.depth(64), self.depth(96), 3, padding=1),
            BasicConv2d(self.depth(96), self.depth(96), 3, padding=1)
        )
        layers['Branch_3'] = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(self.depth(288), self.depth(64), 1)
        )
        return layers

    def _make_Mixed_6a(self):
        layers = nn.ModuleDict()
        layers['Branch_0'] = BasicConv2d(self.depth(288), self.depth(384), 3, stride=2, padding=0)
        layers['Branch_1'] = nn.Sequential(
            BasicConv2d(self.depth(288), self.depth(64), 1),
            BasicConv2d(self.depth(64), self.depth(96), 3, padding=1),
            BasicConv2d(self.depth(96), self.depth(96), 3, stride=2, padding=0)
        )
        layers['Branch_2'] = nn.MaxPool2d(3, stride=2, padding=0)
        return layers

    def _make_Mixed_6b(self):
        layers = nn.ModuleDict()
        layers['Branch_0'] = BasicConv2d(self.depth(768), self.depth(192), 1)
        layers['Branch_1'] = nn.Sequential(
            BasicConv2d(self.depth(768), self.depth(128), 1),
            BasicConv2d(self.depth(128), self.depth(128), (1,7), padding=(0,3)),
            BasicConv2d(self.depth(128), self.depth(192), (7,1), padding=(3,0))
        )
        layers['Branch_2'] = nn.Sequential(
            BasicConv2d(self.depth(768), self.depth(128), 1),
            BasicConv2d(self.depth(128), self.depth(128), (7,1), padding=(3,0)),
            BasicConv2d(self.depth(128), self.depth(128), (1,7), padding=(0,3)),
            BasicConv2d(self.depth(128), self.depth(128), (7,1), padding=(3,0)),
            BasicConv2d(self.depth(128), self.depth(192), (1,7), padding=(0,3))
        )
        layers['Branch_3'] = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(self.depth(768), self.depth(192), 1)
        )
        return layers

    def _make_Mixed_6c(self):
        layers = nn.ModuleDict()
        layers['Branch_0'] = BasicConv2d(self.depth(768), self.depth(192), 1)
        layers['Branch_1'] = nn.Sequential(
            BasicConv2d(self.depth(768), self.depth(160), 1),
            BasicConv2d(self.depth(160), self.depth(160), (1,7), padding=(0,3)),
            BasicConv2d(self.depth(160), self.depth(192), (7,1), padding=(3,0))
        )
        layers['Branch_2'] = nn.Sequential(
            BasicConv2d(self.depth(768), self.depth(160), 1),
            BasicConv2d(self.depth(160), self.depth(160), (7,1), padding=(3,0)),
            BasicConv2d(self.depth(160), self.depth(160), (1,7), padding=(0,3)),
            BasicConv2d(self.depth(160), self.depth(160), (7,1), padding=(3,0)),
            BasicConv2d(self.depth(160), self.depth(192), (1,7), padding=(0,3))
        )
        layers['Branch_3'] = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(self.depth(768), self.depth(192), 1)
        )
        return layers

    def _make_Mixed_6d(self):
        layers = nn.ModuleDict()
        layers['Branch_0'] = BasicConv2d(self.depth(768), self.depth(192), 1)
        layers['Branch_1'] = nn.Sequential(
            BasicConv2d(self.depth(768), self.depth(160), 1),
            BasicConv2d(self.depth(160), self.depth(160), (1,7), padding=(0,3)),
            BasicConv2d(self.depth(160), self.depth(192), (7,1), padding=(3,0))
        )
        layers['Branch_2'] = nn.Sequential(
            BasicConv2d(self.depth(768), self.depth(160), 1),
            BasicConv2d(self.depth(160), self.depth(160), (7,1), padding=(3,0)),
            BasicConv2d(self.depth(160), self.depth(160), (1,7), padding=(0,3)),
            BasicConv2d(self.depth(160), self.depth(160), (7,1), padding=(3,0)),
            BasicConv2d(self.depth(160), self.depth(192), (1,7), padding=(0,3))
        )
        layers['Branch_3'] = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(self.depth(768), self.depth(192), 1)
        )
        return layers

    def _make_Mixed_6e(self):
        layers = nn.ModuleDict()
        layers['Branch_0'] = BasicConv2d(self.depth(768), self.depth(192), 1)
        layers['Branch_1'] = nn.Sequential(
            BasicConv2d(self.depth(768), self.depth(192), 1),
            BasicConv2d(self.depth(192), self.depth(192), (1,7), padding=(0,3)),
            BasicConv2d(self.depth(192), self.depth(192), (7,1), padding=(3,0))
        )
        layers['Branch_2'] = nn.Sequential(
            BasicConv2d(self.depth(768), self.depth(192), 1),
            BasicConv2d(self.depth(192), self.depth(192), (7,1), padding=(3,0)),
            BasicConv2d(self.depth(192), self.depth(192), (1,7), padding=(0,3)),
            BasicConv2d(self.depth(192), self.depth(192), (7,1), padding=(3,0)),
            BasicConv2d(self.depth(192), self.depth(192), (1,7), padding=(0,3))
        )
        layers['Branch_3'] = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(self.depth(768), self.depth(192), 1)
        )
        return layers

    def _make_Mixed_7a(self):
        layers = nn.ModuleDict()
        layers['Branch_0'] = nn.Sequential(
            BasicConv2d(self.depth(768), self.depth(192), 1),
            BasicConv2d(self.depth(192), self.depth(320), 3, stride=2, padding=0)
        )
        layers['Branch_1'] = nn.Sequential(
            BasicConv2d(self.depth(768), self.depth(192), 1),
            BasicConv2d(self.depth(192), self.depth(192), (1,7), padding=(0,3)),
            BasicConv2d(self.depth(192), self.depth(192), (7,1), padding=(3,0)),
            BasicConv2d(self.depth(192), self.depth(192), 3, stride=2, padding=0)
        )
        layers['Branch_2'] = nn.MaxPool2d(3, stride=2, padding=0)
        return layers

    def _make_Mixed_7b(self):
        layers = nn.ModuleDict()
        layers['Branch_0'] = BasicConv2d(self.depth(1280), self.depth(320), 1)
        layers['Branch_1'] = nn.Sequential(
            BasicConv2d(self.depth(1280), self.depth(384), 1),
            nn.Concat(dim=1),
            BasicConv2d(self.depth(384), self.depth(384), (1,3), padding=(0,1)),
            BasicConv2d(self.depth(384), self.depth(384), (3,1), padding=(1,0))
        )
        layers['Branch_2'] = nn.Sequential(
            BasicConv2d(self.depth(1280), self.depth(448), 1),
            BasicConv2d(self.depth(448), self.depth(384), 3, padding=1),
            nn.Concat(dim=1),
            BasicConv2d(self.depth(384), self.depth(384), (1,3), padding=(0,1)),
            BasicConv2d(self.depth(384), self.depth(384), (3,1), padding=(1,0))
        )
        layers['Branch_3'] = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(self.depth(1280), self.depth(192), 1)
        )
        return layers

    def _make_Mixed_7c(self):
        layers = nn.ModuleDict()
        layers['Branch_0'] = BasicConv2d(self.depth(2048), self.depth(320), 1)
        layers['Branch_1'] = nn.Sequential(
            BasicConv2d(self.depth(2048), self.depth(384), 1),
            nn.Concat(dim=1),
            BasicConv2d(self.depth(384), self.depth(384), (1,3), padding=(0,1)),
            BasicConv2d(self.depth(384), self.depth(384), (3,1), padding=(1,0))
        )
        layers['Branch_2'] = nn.Sequential(
            BasicConv2d(self.depth(2048), self.depth(448), 1),
            BasicConv2d(self.depth(448), self.depth(384), 3, padding=1),
            nn.Concat(dim=1),
            BasicConv2d(self.depth(384), self.depth(384), (1,3), padding=(0,1)),
            BasicConv2d(self.depth(384), self.depth(384), (3,1), padding=(1,0))
        )
        layers['Branch_3'] = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(self.depth(2048), self.depth(192), 1)
        )
        return layers

    def forward(self, x):
        end_points = {}
        
        x = self.Conv2d_1a_3x3(x)
        end_points['Conv2d_1a_3x3'] = x
        
        x = self.Conv2d_2a_3x3(x)
        end_points['Conv2d_2a_3x3'] = x
        
        x = self.Conv2d_2b_3x3(x)
        end_points['Conv2d_2b_3x3'] = x
        
        x = self.MaxPool_3a_3x3(x)
        end_points['MaxPool_3a_3x3'] = x
        
        x = self.Conv2d_3b_1x1(x)
        end_points['Conv2d_3b_1x1'] = x
        
        x = self.Conv2d_4a_3x3(x)
        end_points['Conv2d_4a_3x3'] = x
        
        x = self.MaxPool_5a_3x3(x)
        end_points['MaxPool_5a_3x3'] = x

        branch_0 = self.Mixed_5b['Branch_0'](x)
        branch_1 = self.Mixed_5b['Branch_1'](x)
        branch_2 = self.Mixed_5b['Branch_2'](x)
        branch_3 = self.Mixed_5b['Branch_3'](x)
        x = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)
        end_points['Mixed_5b'] = x

        branch_0 = self.Mixed_5c['Branch_0'](x)
        branch_1 = self.Mixed_5c['Branch_1'](x)
        branch_2 = self.Mixed_5c['Branch_2'](x)
        branch_3 = self.Mixed_5c['Branch_3'](x)
        x = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)
        end_points['Mixed_5c'] = x

        branch_0 = self.Mixed_5d['Branch_0'](x)
        branch_1 = self.Mixed_5d['Branch_1'](x)
        branch_2 = self.Mixed_5d['Branch_2'](x)
        branch_3 = self.Mixed_5d['Branch_3'](x)
        x = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)
        end_points['Mixed_5d'] = x

        branch_0 = self.Mixed_6a['Branch_0'](x)
        branch_1 = self.Mixed_6a['Branch_1'](x)
        branch_2 = self.Mixed_6a['Branch_2'](x)
        x = torch.cat([branch_0, branch_1, branch_2], 1)
        end_points['Mixed_6a'] = x

        branch_0 = self.Mixed_6b['Branch_0'](x)
        branch_1 = self.Mixed_6b['Branch_1'](x)
        branch_2 = self.Mixed_6b['Branch_2'](x)
        branch_3 = self.Mixed_6b['Branch_3'](x)
        x = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)
        end_points['Mixed_6b'] = x

        branch_0 = self.Mixed_6c['Branch_0'](x)
        branch_1 = self.Mixed_6c['Branch_1'](x)
        branch_2 = self.Mixed_6c['Branch_2'](x)
        branch_3 = self.Mixed_6c['Branch_3'](x)
        x = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)
        end_points['Mixed_6c'] = x

        branch_0 = self.Mixed_6d['Branch_0'](x)
        branch_1 = self.Mixed_6d['Branch_1'](x)
        branch_2 = self.Mixed_6d['Branch_2'](x)
        branch_3 = self.Mixed_6d['Branch_3'](x)
        x = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)
        end_points['Mixed_6d'] = x

        branch_0 = self.Mixed_6e['Branch_0'](x)
        branch_1 = self.Mixed_6e['Branch_1'](x)
        branch_2 = self.Mixed_6e['Branch_2'](x)
        branch_3 = self.Mixed_6e['Branch_3'](x)
        x = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)
        end_points['Mixed_6e'] = x

        branch_0 = self.Mixed_7a['Branch_0'](x)
        branch_1 = self.Mixed_7a['Branch_1'](x)
        branch_2 = self.Mixed_7a['Branch_2'](x)
        x = torch.cat([branch_0, branch_1, branch_2], 1)
        end_points['Mixed_7a'] = x

        branch_0 = self.Mixed_7b['Branch_0'](x)
        branch_1 = self.Mixed_7b['Branch_1'][0](x)
        branch_1 = torch.cat([self.Mixed_7b['Branch_1'][2](branch_1), self.Mixed_7b['Branch_1'][3](branch_1)], 1)
        branch_2 = self.Mixed_7b['Branch_2'][0](x)
        branch_2 = self.Mixed_7b['Branch_2'][1](branch_2)
        branch_2 = torch.cat([self.Mixed_7b['Branch_2'][3](branch_2), self.Mixed_7b['Branch_2'][4](branch_2)], 1)
        branch_3 = self.Mixed_7b['Branch_3'](x)
        x = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)
        end_points['Mixed_7b'] = x

        branch_0 = self.Mixed_7c['Branch_0'](x)
        branch_1 = self.Mixed_7c['Branch_1'][0](x)
        branch_1 = torch.cat([self.Mixed_7c['Branch_1'][2](branch_1), self.Mixed_7c['Branch_1'][3](branch_1)], 1)
        branch_2 = self.Mixed_7c['Branch_2'][0](x)
        branch_2 = self.Mixed_7c['Branch_2'][1](branch_2)
        branch_2 = torch.cat([self.Mixed_7c['Branch_2'][3](branch_2), self.Mixed_7c['Branch_2'][4](branch_2)], 1)
        branch_3 = self.Mixed_7c['Branch_3'](x)
        x = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)
        end_points['Mixed_7c'] = x

        return x, end_points

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, min_depth=16, depth_multiplier=1.0, create_aux_logits=True, dropout_keep_prob=0.8):
        super(InceptionV3, self).__init__()
        self.create_aux_logits = create_aux_logits
        self.dropout_keep_prob = dropout_keep_prob
        self.base = InceptionV3Base(min_depth=min_depth, depth_multiplier=depth_multiplier)
        self.depth = lambda d: max(int(d * depth_multiplier), min_depth)

        if self.create_aux_logits:
            self.AuxLogits = nn.Sequential(
                nn.AvgPool2d(5, stride=3, padding=0),
                BasicConv2d(self.depth(768), self.depth(128), 1),
                BasicConv2d(self.depth(128), self.depth(768), 5, padding=0, bias=True),
                nn.Conv2d(self.depth(768), num_classes, 1, bias=True)
            )

        self.Logits = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(1 - self.dropout_keep_prob),
            nn.Conv2d(self.depth(2048), num_classes, 1, bias=True)
        )

    def forward(self, x):
        x, end_points = self.base(x)

        if self.create_aux_logits and self.training:
            aux_logits = self.AuxLogits(end_points['Mixed_6e'])
            aux_logits = aux_logits.squeeze(2).squeeze(2)
            end_points['AuxLogits'] = aux_logits

        logits = self.Logits(x)
        logits = logits.squeeze(2).squeeze(2)
        end_points['Logits'] = logits
        end_points['Predictions'] = F.softmax(logits, dim=1)

        if self.create_aux_logits and self.training:
            return logits, aux_logits, end_points
        else:
            return logits, end_points

if __name__ == '__main__':
    model = InceptionV3(num_classes=1000)
    model.eval()
    x = torch.randn(1, 3, 299, 299)
    logits, end_points = model(x)
    print(logits.shape)