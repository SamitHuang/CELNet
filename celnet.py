import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import model.ops as ops 

class CELNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, num_filters=16, head_max_pooling=True, attention="mam", companion_output=False):
        self.inplanes = num_filters 
        fm_size= 12 if head_max_pooling else 24

        super(CELNet, self).__init__()
        self.conv1 =  nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        if head_max_pooling:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = None
        self.layer1 = self._make_layer(block, num_filters, layers[0], attention=attention)
        self.layer2 = self._make_layer(block, num_filters*2, layers[1], stride=2, attention=attention)
        self.layer3 = self._make_layer(block, num_filters*4, layers[2], stride=2, attention=attention)

        # compainion output 
        self.companion_output = companion_output
        if head_max_pooling:
            # TODO: the layer output size should be computed automatically based on the input and downsample stream.  
            layer2_output_size = 24 
            layer3_output_size = 12
        else:
            layer2_output_size = 48 
            layer3_output_size = 24 
        self.layer3_output_size = layer3_output_size

        if companion_output:
            self.avgpool_l2= nn.AvgPool2d(layer2_output_size)
            self.maxpool_l2 = nn.MaxPool2d(layer2_output_size)
            self.avgpool_l3= nn.AvgPool2d(layer3_output_size)
            self.maxpool_l3 = nn.MaxPool2d(layer3_output_size)
 
        # FC layers or FCN layers 
        if companion_output:
            companion_output_size = 2*(64+32) 
            self.classifer = nn.Sequential(
                    #nn.BatchNorm1d(companion_output_size),
                    #nn.Dropout(0.2),
                    #nn.Linear(companion_output_size, 256),
                    #nn.ReLU(),
                    #nn.BatchNorm1d(256),
                    #nn.Dropout(0.2),
                    nn.Linear(companion_output_size, 256),
                    nn.Linear(256, num_classes)
                    )
        else:
            self.classifer = nn.Linear(num_filters * 4 * block.expansion, num_classes)
        #self.avgpool =  nn.AvgPool2d(fm_size, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, attention='mam'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, attention))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention=attention))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool != None:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if self.companion_output:
            l2_companion_output = torch.cat((self.avgpool_l2(x), self.maxpool_l2(x)), dim=1) 
        x = self.layer3(x)
        if self.companion_output:
            l3_companion_output = torch.cat((self.avgpool_l3(x), self.maxpool_l3(x)), dim=1) 

        if self.companion_output:
            x = torch.cat((l2_companion_output,l3_companion_output), dim=1)
            x = x.view(x.size(0), -1)
            x = self.classifer(x)
        else:
            x = self.avgpool_l3(x)
            x = nn.AvgPool2d(self.layer3_output_size)
            x = x.view(x.size(0), -1)
            x = self.classifer(x)
        
        return x

def celnet(head_max_pooling=False, attention="mam", companion_output=True):
    model = CELNet(BasicBlock, [3, 3, 3], num_classes=1, num_filters=16, head_max_pooling=head_max_pooling, attention=attention, companion_output=companion_output)
    return model

def conv3x3(in_planes, out_planes, stride=1):
    #"3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3, multi_branch=False):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        #self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.multi_branch = multi_branch
        if multi_branch:
            self.conv2 = nn.Conv2d(2, 1, 5, padding=2, bias=False) 
            self.conv3 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        if not self.multi_branch:
            x = self.conv1(x)
        else:
            # multi branch, 3x3, 5x5, 7x7
            x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention='mam'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if attention == 'se':
            self.ca = ChannelAttention(planes, ratio=4)
            self.sa = None
        elif attention =='cbam':
            self.ca = ChannelAttention(planes, ratio=4)
            self.sa = SpatialAttention()
        elif attention =='mam':
            self.ca = ChannelAttention(planes, ratio=4)
            self.sa = SpatialAttention(multi_branch=True)
        elif attention == None:
            self.ca = None
            self.sa = None

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.ca != None: 
            out = self.ca(out) * out
        if self.sa != None:
            out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ConcatAvgMaxPooling(nn.Module):
    # x shape, [bs, c, h, w]
    def __init__(self,kernel_size=12, stride=1):
        super(ConcatAvgMaxPooling, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride=1) 
        self.maxpool = nn.MaxPool2d(kernel_size, stride=1)

    def forward(self, x):
        x = torch.cat((self.avgpool(x), self.maxpool(x)), axis=1)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
