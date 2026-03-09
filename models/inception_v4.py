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

def block_inception_a(inputs):
    branch_0 = BasicConv2d(inputs.size(1), 96, 1)(inputs)
    
    branch_1 = BasicConv2d(inputs.size(1), 64, 1)(inputs)
    branch_1 = BasicConv2d(64, 96, 3, padding=1)(branch_1)
    
    branch_2 = BasicConv2d(inputs.size(1), 64, 1)(inputs)
    branch_2 = BasicConv2d(64, 96, 3, padding=1)(branch_2)
    branch_2 = BasicConv2d(96, 96, 3, padding=1)(branch_2)
    
    branch_3 = nn.AvgPool2d(3, stride=1, padding=1)(inputs)
    branch_3 = BasicConv2d(inputs.size(1), 96, 1)(branch_3)
    
    return torch.cat([branch_0, branch_1, branch_2, branch_3], 1)

def block_reduction_a(inputs):
    branch_0 = BasicConv2d(inputs.size(1), 384, 3, stride=2, padding=0)(inputs)
    
    branch_1 = BasicConv2d(inputs.size(1), 192, 1)(inputs)
    branch_1 = BasicConv2d(192, 224, 3, padding=1)(branch_1)
    branch_1 = BasicConv2d(224, 256, 3, stride=2, padding=0)(branch_1)
    
    branch_2 = nn.MaxPool2d(3, stride=2, padding=0)(inputs)
    
    return torch.cat([branch_0, branch_1, branch_2], 1)

def block_inception_b(inputs):
    branch_0 = BasicConv2d(inputs.size(1), 384, 1)(inputs)
    
    branch_1 = BasicConv2d(inputs.size(1), 192, 1)(inputs)
    branch_1 = BasicConv2d(192, 224, (1,7), padding=(0,3))(branch_1)
    branch_1 = BasicConv2d(224, 256, (7,1), padding=(3,0))(branch_1)
    
    branch_2 = BasicConv2d(inputs.size(1), 192, 1)(inputs)
    branch_2 = BasicConv2d(192, 192, (7,1), padding=(3,0))(branch_2)
    branch_2 = BasicConv2d(192, 224, (1,7), padding=(0,3))(branch_2)
    branch_2 = BasicConv2d(224, 224, (7,1), padding=(3,0))(branch_2)
    branch_2 = BasicConv2d(224, 256, (1,7), padding=(0,3))(branch_2)
    
    branch_3 = nn.AvgPool2d(3, stride=1, padding=1)(inputs)
    branch_3 = BasicConv2d(inputs.size(1), 128, 1)(branch_3)
    
    return torch.cat([branch_0, branch_1, branch_2, branch_3], 1)

def block_reduction_b(inputs):
    branch_0 = BasicConv2d(inputs.size(1), 192, 1)(inputs)
    branch_0 = BasicConv2d(192, 192, 3, stride=2, padding=0)(branch_0)
    
    branch_1 = BasicConv2d(inputs.size(1), 256, 1)(inputs)
    branch_1 = BasicConv2d(256, 256, (1,7), padding=(0,3))(branch_1)
    branch_1 = BasicConv2d(256, 320, (7,1), padding=(3,0))(branch_1)
    branch_1 = BasicConv2d(320, 320, 3, stride=2, padding=0)(branch_1)
    
    branch_2 = nn.MaxPool2d(3, stride=2, padding=0)(inputs)
    
    return torch.cat([branch_0, branch_1, branch_2], 1)

def block_inception_c(inputs):
    branch_0 = BasicConv2d(inputs.size(1), 256, 1)(inputs)
    
    branch_1 = BasicConv2d(inputs.size(1), 384, 1)(inputs)
    branch_1_1 = BasicConv2d(384, 256, (1,3), padding=(0,1))(branch_1)
    branch_1_2 = BasicConv2d(384, 256, (3,1), padding=(1,0))(branch_1)
    branch_1 = torch.cat([branch_1_1, branch_1_2], 1)
    
    branch_2 = BasicConv2d(inputs.size(1), 384, 1)(inputs)
    branch_2 = BasicConv2d(384, 448, (3,1), padding=(1,0))(branch_2)
    branch_2 = BasicConv2d(448, 512, (1,3), padding=(0,1))(branch_2)
    branch_2_1 = BasicConv2d(512, 256, (1,3), padding=(0,1))(branch_2)
    branch_2_2 = BasicConv2d(512, 256, (3,1), padding=(1,0))(branch_2)
    branch_2 = torch.cat([branch_2_1, branch_2_2], 1)
    
    branch_3 = nn.AvgPool2d(3, stride=1, padding=1)(inputs)
    branch_3 = BasicConv2d(inputs.size(1), 256, 1)(branch_3)
    
    return torch.cat([branch_0, branch_1, branch_2, branch_3], 1)

class InceptionV4Base(nn.Module):
    def __init__(self):
        super(InceptionV4Base, self).__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=0)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, padding=1)

    def forward(self, x, final_endpoint='Mixed_7d'):
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
        
        with torch.no_grad():
            branch_0 = nn.MaxPool2d(3, stride=2, padding=0)(net)
            branch_1 = BasicConv2d(64, 96, 3, stride=2, padding=0)(net)
            net = torch.cat([branch_0, branch_1], 1)
        if add_and_check_final('Mixed_3a', net): return net, end_points
        
        branch_0 = BasicConv2d(160, 64, 1)(net)
        branch_0 = BasicConv2d(64, 96, 3, padding=0)(branch_0)
        branch_1 = BasicConv2d(160, 64, 1)(net)
        branch_1 = BasicConv2d(64, 64, (1,7), padding=(0,3))(branch_1)
        branch_1 = BasicConv2d(64, 64, (7,1), padding=(3,0))(branch_1)
        branch_1 = BasicConv2d(64, 96, 3, padding=0)(branch_1)
        net = torch.cat([branch_0, branch_1], 1)
        if add_and_check_final('Mixed_4a', net): return net, end_points
        
        branch_0 = BasicConv2d(192, 192, 3, stride=2, padding=0)(net)
        branch_1 = nn.MaxPool2d(3, stride=2, padding=0)(net)
        net = torch.cat([branch_0, branch_1], 1)
        if add_and_check_final('Mixed_5a', net): return net, end_points
        
        for idx in range(4):
            block_scope = 'Mixed_5' + chr(ord('b') + idx)
            net = block_inception_a(net)
            if add_and_check_final(block_scope, net): return net, end_points
        
        net = block_reduction_a(net)
        if add_and_check_final('Mixed_6a', net): return net, end_points
        
        for idx in range(7):
            block_scope = 'Mixed_6' + chr(ord('b') + idx)
            net = block_inception_b(net)
            if add_and_check_final(block_scope, net): return net, end_points
        
        net = block_reduction_b(net)
        if add_and_check_final('Mixed_7a', net): return net, end_points
        
        for idx in range(3):
            block_scope = 'Mixed_7' + chr(ord('b') + idx)
            net = block_inception_c(net)
            if add_and_check_final(block_scope, net): return net, end_points
        
        raise ValueError('Unknown final endpoint %s' % final_endpoint)

class InceptionV4(nn.Module):
    def __init__(self, num_classes=1001, dropout_keep_prob=0.8, create_aux_logits=True):
        super(InceptionV4, self).__init__()
        self.base = InceptionV4Base()
        self.dropout_keep_prob = dropout_keep_prob
        self.create_aux_logits = create_aux_logits
        
        if self.create_aux_logits:
            self.AuxLogits = nn.Sequential(
                nn.AvgPool2d(5, stride=3, padding=0),
                BasicConv2d(1024, 128, 1),
                nn.Flatten(),
                nn.Linear(128 * 5 * 5, 768),
                nn.ReLU(inplace=True),
                nn.Linear(768, num_classes)
            )
        
        self.Logits = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(1 - self.dropout_keep_prob),
            nn.Flatten(),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        end_points = {}
        net, end_points = self.base(x)
        
        if self.create_aux_logits and self.training:
            aux_logits = self.AuxLogits(end_points['Mixed_6h'])
            end_points['AuxLogits'] = aux_logits
        
        logits = self.Logits(net)
        end_points['Logits'] = logits
        end_points['Predictions'] = F.softmax(logits, dim=1)
        
        if self.create_aux_logits and self.training:
            return logits, aux_logits, end_points
        else:
            return logits, end_points

if __name__ == '__main__':
    model = InceptionV4(num_classes=1001)
    model.eval()
    x = torch.randn(1, 3, 299, 299)
    logits, end_points = model(x)
    print(logits.shape)