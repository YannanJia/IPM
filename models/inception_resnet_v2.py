import torch
import torch.nn as nn
import torch.nn.functional as F

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

def block35(net, scale=1.0, activation_fn=F.relu):
    in_channels = net.size(1)
    
    tower_conv = BasicConv2d(in_channels, 32, 1)(net)
    
    tower_conv1_0 = BasicConv2d(in_channels, 32, 1)(net)
    tower_conv1_1 = BasicConv2d(32, 32, 3, padding=1)(tower_conv1_0)
    
    tower_conv2_0 = BasicConv2d(in_channels, 32, 1)(net)
    tower_conv2_1 = BasicConv2d(32, 48, 3, padding=1)(tower_conv2_0)
    tower_conv2_2 = BasicConv2d(48, 64, 3, padding=1)(tower_conv2_1)
    
    mixed = torch.cat([tower_conv, tower_conv1_1, tower_conv2_2], 1)
    up = nn.Conv2d(mixed.size(1), in_channels, 1, bias=True)(mixed)
    net += scale * up
    
    if activation_fn is not None:
        net = activation_fn(net)
    return net

def block17(net, scale=1.0, activation_fn=F.relu):
    in_channels = net.size(1)
    
    tower_conv = BasicConv2d(in_channels, 192, 1)(net)
    
    tower_conv1_0 = BasicConv2d(in_channels, 128, 1)(net)
    tower_conv1_1 = BasicConv2d(128, 160, (1,7), padding=(0,3))(tower_conv1_0)
    tower_conv1_2 = BasicConv2d(160, 192, (7,1), padding=(3,0))(tower_conv1_1)
    
    mixed = torch.cat([tower_conv, tower_conv1_2], 1)
    up = nn.Conv2d(mixed.size(1), in_channels, 1, bias=True)(mixed)
    net += scale * up
    
    if activation_fn is not None:
        net = activation_fn(net)
    return net

def block8(net, scale=1.0, activation_fn=F.relu):
    in_channels = net.size(1)
    
    tower_conv = BasicConv2d(in_channels, 192, 1)(net)
    
    tower_conv1_0 = BasicConv2d(in_channels, 192, 1)(net)
    tower_conv1_1 = BasicConv2d(192, 224, (1,3), padding=(0,1))(tower_conv1_0)
    tower_conv1_2 = BasicConv2d(224, 256, (3,1), padding=(1,0))(tower_conv1_1)
    
    mixed = torch.cat([tower_conv, tower_conv1_2], 1)
    up = nn.Conv2d(mixed.size(1), in_channels, 1, bias=True)(mixed)
    net += scale * up
    
    if activation_fn is not None:
        net = activation_fn(net)
    return net

class InceptionResNetV2Base(nn.Module):
    def __init__(self, output_stride=16, align_feature_maps=False):
        super(InceptionResNetV2Base, self).__init__()
        self.output_stride = output_stride
        self.padding = 1 if align_feature_maps else 0
        
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=self.padding)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, padding=self.padding)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, padding=1)
        self.MaxPool_3a_3x3 = nn.MaxPool2d(3, stride=2, padding=self.padding)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1, padding=self.padding)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3, padding=self.padding)
        self.MaxPool_5a_3x3 = nn.MaxPool2d(3, stride=2, padding=self.padding)

    def forward(self, x, final_endpoint='Conv2d_7b_1x1'):
        end_points = {}
        
        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint
        
        net = self.Conv2d_1a_3x3(x)
        if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points
        
        net = self.Conv2d_2a_3x3(net)
        if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
        
        net = self.Conv2d_2b_3x3(net)
        if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
        
        net = self.MaxPool_3a_3x3(net)
        if add_and_check_final('MaxPool_3a_3x3', net): return net, end_points
        
        net = self.Conv2d_3b_1x1(net)
        if add_and_check_final('Conv2d_3b_1x1', net): return net, end_points
        
        net = self.Conv2d_4a_3x3(net)
        if add_and_check_final('Conv2d_4a_3x3', net): return net, end_points
        
        net = self.MaxPool_5a_3x3(net)
        if add_and_check_final('MaxPool_5a_3x3', net): return net, end_points
        
        tower_conv = BasicConv2d(192, 96, 1)(net)
        tower_conv1_0 = BasicConv2d(192, 48, 1)(net)
        tower_conv1_1 = BasicConv2d(48, 64, 5, padding=2)(tower_conv1_0)
        tower_conv2_0 = BasicConv2d(192, 64, 1)(net)
        tower_conv2_1 = BasicConv2d(64, 96, 3, padding=1)(tower_conv2_0)
        tower_conv2_2 = BasicConv2d(96, 96, 3, padding=1)(tower_conv2_1)
        tower_pool = nn.AvgPool2d(3, stride=1, padding=1)(net)
        tower_pool_1 = BasicConv2d(192, 64, 1)(tower_pool)
        net = torch.cat([tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 1)
        
        if add_and_check_final('Mixed_5b', net): return net, end_points
        
        for _ in range(10):
            net = block35(net, scale=0.17)
        
        use_atrous = self.output_stride == 8
        stride = 1 if use_atrous else 2
        
        tower_conv = BasicConv2d(320, 384, 3, stride=stride, padding=self.padding)(net)
        tower_conv1_0 = BasicConv2d(320, 256, 1)(net)
        tower_conv1_1 = BasicConv2d(256, 256, 3, padding=1)(tower_conv1_0)
        tower_conv1_2 = BasicConv2d(256, 384, 3, stride=stride, padding=self.padding)(tower_conv1_1)
        tower_pool = nn.MaxPool2d(3, stride=stride, padding=self.padding)(net)
        net = torch.cat([tower_conv, tower_conv1_2, tower_pool], 1)
        
        if add_and_check_final('Mixed_6a', net): return net, end_points
        
        for _ in range(20):
            net = block17(net, scale=0.10)
        
        if add_and_check_final('PreAuxLogits', net): return net, end_points
        
        if self.output_stride == 8:
            raise ValueError('output_stride==8 is only supported up to the PreAuxlogits end_point for now.')
        
        tower_conv = BasicConv2d(1088, 256, 1)(net)
        tower_conv_1 = BasicConv2d(256, 384, 3, stride=2, padding=self.padding)(tower_conv)
        tower_conv1 = BasicConv2d(1088, 256, 1)(net)
        tower_conv1_1 = BasicConv2d(256, 288, 3, stride=2, padding=self.padding)(tower_conv1)
        tower_conv2 = BasicConv2d(1088, 256, 1)(net)
        tower_conv2_1 = BasicConv2d(256, 288, 3, padding=1)(tower_conv2)
        tower_conv2_2 = BasicConv2d(288, 320, 3, stride=2, padding=self.padding)(tower_conv2_1)
        tower_pool = nn.MaxPool2d(3, stride=2, padding=self.padding)(net)
        net = torch.cat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 1)
        
        if add_and_check_final('Mixed_7a', net): return net, end_points
        
        for _ in range(9):
            net = block8(net, scale=0.20)
        net = block8(net, activation_fn=None)
        
        net = BasicConv2d(net.size(1), 1536, 1)(net)
        if add_and_check_final('Conv2d_7b_1x1', net): return net, end_points
        
        raise ValueError(f'final_endpoint ({final_endpoint}) not recognized')

class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=1001, dropout_keep_prob=0.8, create_aux_logits=True, output_stride=16, align_feature_maps=False):
        super(InceptionResNetV2, self).__init__()
        self.base = InceptionResNetV2Base(output_stride=output_stride, align_feature_maps=align_feature_maps)
        self.dropout_keep_prob = dropout_keep_prob
        self.create_aux_logits = create_aux_logits
        
        if self.create_aux_logits:
            self.AuxLogits = nn.Sequential(
                nn.AvgPool2d(5, stride=3, padding=0),
                BasicConv2d(1088, 128, 1),
                nn.Flatten(),
                nn.Linear(128 * 5 * 5, 768),
                nn.ReLU(inplace=True),
                nn.Linear(768, num_classes)
            )
        
        self.Logits = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(1 - self.dropout_keep_prob),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        end_points = {}
        net, end_points = self.base(x)
        
        if self.create_aux_logits and self.training:
            aux_logits = self.AuxLogits(end_points['PreAuxLogits'])
            end_points['AuxLogits'] = aux_logits
        
        logits = self.Logits(net)
        end_points['Logits'] = logits
        end_points['Predictions'] = F.softmax(logits, dim=1)
        
        if self.create_aux_logits and self.training:
            return logits, aux_logits, end_points
        else:
            return logits, end_points

if __name__ == '__main__':
    model = InceptionResNetV2(num_classes=1001)
    model.eval()
    x = torch.randn(1, 3, 299, 299)
    logits, end_points = model(x)
    print(logits.shape)