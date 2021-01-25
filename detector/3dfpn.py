'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import math
from torchsummary import summary          #使用 pip install torchsummary

config = {}
config['anchors'] = [5., 10., 20.] #[ 10.0, 30.0, 60.]
config['chanel'] = 1
config['crop_size'] = [96, 96, 96]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 10 #2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 2.5 #3 #6. #mm
config['sizelim2'] = 10 #30
config['sizelim3'] = 20 #40
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','990fbe3f0a1b53878669967b9afd1441','adc3bbc63d40f8761c59be10f1e504c3']
debug = True #True#False #True


## Backbone
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
'''

def init_conv_weights(layer, weights_std=0.01,  bias=0):
    '''
    RetinaNet's layer initialization

    :layer
    :

    '''
    nn.init.xavier_normal(layer.weight)
    nn.init.constant(layer.bias.data, val=bias)
    return layer

def conv1x1x1(in_channels, out_channels, **kwargs):
    '''Return a 1x1 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv3d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)

    return layer

def conv3x3x3(in_channels, out_channels, **kwargs):
    '''Return a 3x3 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv3d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)

    return layer

'''
def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #conv2_rep = out
        out = self.bn2(out) #Out: [1, 64, 20, 20, 20]
        # print(out.shape)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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
        #conv3_rep = out
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        '''
        Args:
        block: BasicBlock for resnet18,34;  Bottleneck for resnet50,101,134
        layers: number of repeated layers in four conv region, eg. [2,2,2,2] for resnet18
        '''
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=2,   
                               bias=False)             # 128 -> 64
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)    # 64 -> 32
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)    # 32 -> 16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)    # 16 -> 8
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)    # 8 -> 4
        '''
        self.avgpool = nn.AvgPool3d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        '''

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.xavier_normal(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):       # 128 
        c1 = self.conv1(x)       # 64 --> 8 anchor_area
        c1 = self.bn1(c1)         
        c1 = self.relu(c1)
        c2 = self.maxpool(c1)     # 32

        c2 = self.layer1(c2)  # 32 --> 16 anchor_area
        c3 = self.layer2(c2)  # 16 --> 32 anchor_area
        c4 = self.layer3(c3)  # 8
        c5 = self.layer4(c4)  # 4
        '''
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        '''
        return c1, c2, c3, c4, c5


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class FeaturePyramid(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid, self).__init__()
        self.resnet = resnet

        # applied in a pyramid
        # self.pyramid_transformation_1 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_1 = conv3x3x3(64, 256)
        self.pyramid_transformation_2 = conv1x1x1(64, 64)
        self.pyramid_transformation_3 = conv1x1x1(128, 128)
        self.pyramid_transformation_4 = conv1x1x1(256, 256)
        self.pyramid_transformation_5 = conv1x1x1(512, 512)

        # applied after upsampling and before adding to calculate pyramid_feature_2
        self.pyramid_transformation_512_to_64 = conv1x1x1(512, 64)
        self.pyramid_transformation_256_to_64 = conv1x1x1(256, 64)
        self.pyramid_transformation_128_to_64 = conv1x1x1(128, 64)

        # applied after upsampling and before adding to calculate pyramid_feature_3
        self.pyramid_transformation_512_to_128 = conv1x1x1(512, 128)
        self.pyramid_transformation_256_to_128 = conv1x1x1(256, 128)
        self.pyramid_transformation_64_to_128 = conv1x1x1(64, 128)
        
        # applied after upsampling and before adding to calculate pyramid_feature_4
        self.pyramid_transformation_512_to_256 = conv1x1x1(512, 256)
        self.pyramid_transformation_128_to_256 = conv1x1x1(128, 256)
        self.pyramid_transformation_64_to_256 = conv1x1x1(64, 256)

        # applied after upsampling and before adding to calculate pyramid_feature_5
        self.pyramid_transformation_256_to_512 = conv1x1x1(256, 512)
        self.pyramid_transformation_128_to_512 = conv1x1x1(128, 512)
        self.pyramid_transformation_64_to_512 = conv1x1x1(64, 512)

        # applied downsample
        self.downsample_c2_to_c5 = self._downsample(8)
        self.downsample_c3_to_c5 = self._downsample(4)
        self.downsample_c4_to_c5 = self._downsample(2)
        self.downsample_c2_to_c4 = self._downsample(4)
        self.downsample_c3_to_c4 = self._downsample(2)
        self.downsample_c2_to_c3 = self._downsample(2)

        # applied after gathering
        self.upsample_transform_2 = conv3x3x3(64, 256, padding=1)
        self.upsample_transform_3 = conv3x3x3(128, 256, padding=1)
        self.upsample_transform_4 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_5 = conv3x3x3(512, 256, padding=1)


    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        depth, height, width = scaled_feature.size()[2:]
        return F.upsample(original_feature, scale_factor=scale_factor)[:, :, :depth, :height, :width]

    def _downsample(self, scale_factor=2):
        return nn.MaxPool3d(3, stride=scale_factor, padding=1)

    def forward(self, x):
        resnet_feature_1, resnet_feature_2, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)
        # print('resnet_feature_4 shape:', resnet_feature_4.shape)
        # print('resnet_feature_5 shape:', resnet_feature_5.shape)
        # resnet_feature_1: [1, 64, 48, 48, 48]
        # resnet_feature_2: [1, 64, 24, 24, 24]
        # resnet_feature_3: [1, 128, 12, 12, 12]
        # resnet_feature_4: [1, 256, 6, 6, 6] for resnet34, [1, 1024, 6, 6, 6] for resnet50+
        # resnet_feature_5: [1, 512, 3, 3, 3] for resnet34, [1, 2048, 6, 6, 6] for resnet50+
        # print("input:", x.shape)
        # print("resnet_feature_4:", resnet_feature_5.shape)
        # print("resnet_feature_5:", resnet_feature_5.shape)
        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)     # transform c5 to pyramid_c5
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)     # transform c4 to pyramid_c4
        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)     # transform c3 to pyramid_c3
        pyramid_feature_2 = self.pyramid_transformation_2(resnet_feature_2)     # transform c2 to pyramid_c2
       
        # calculate pyramid_feature_2
        upsampled_feature_5_to_2 = self._upsample(pyramid_feature_5, pyramid_feature_2, scale_factor=8)   # deconv c5 to c2.size
        upsampled_feature_4_to_2 = self._upsample(pyramid_feature_4, pyramid_feature_2, scale_factor=4)   # deconv c4 to c2.size
        upsampled_feature_3_to_2 = self._upsample(pyramid_feature_3, pyramid_feature_2, scale_factor=2)   # deconv c3 to c2.size
        pyramid_feature_2 = torch.add(pyramid_feature_2, self.pyramid_transformation_512_to_64(upsampled_feature_5_to_2)) # add deconv_c5 to pyramid_c2
        pyramid_feature_2 = torch.add(pyramid_feature_2, self.pyramid_transformation_256_to_64(upsampled_feature_4_to_2)) # add deconv_c4 to pyramid_c2
        pyramid_feature_2 = torch.add(pyramid_feature_2, self.pyramid_transformation_128_to_64(upsampled_feature_3_to_2)) # add deconv_c3 to pyramid_c2
        pyramid_feature_2 = self.upsample_transform_2(pyramid_feature_2)

        # Calculate pyramid_feature_3
        upsampled_feature_5_to_3 = self.pyramid_transformation_512_to_128(self._upsample(pyramid_feature_5, pyramid_feature_3, scale_factor=4))   # deconv c5 to c3.size
        upsampled_feature_4_to_3 = self.pyramid_transformation_256_to_128(self._upsample(pyramid_feature_4, pyramid_feature_3, scale_factor=2))   # deconv c5 to c3.size
        downsampled_feature_2_to_3 = self.pyramid_transformation_64_to_128(self.downsample_c2_to_c3(resnet_feature_2))                            # downsample c2 to c3 size
        pyramid_feature_3 = torch.add(pyramid_feature_3, upsampled_feature_5_to_3) # add deconv_c5 to pyramid_c3
        pyramid_feature_3 = torch.add(pyramid_feature_3, upsampled_feature_4_to_3) # add deconv_c4 to pyramid_c3
        pyramid_feature_3 = torch.add(pyramid_feature_3, downsampled_feature_2_to_3) # add maxpool_c2 to pyramid_c3
        pyramid_feature_3 = self.upsample_transform_3(pyramid_feature_3)

        # Calculate pyramid_feature_4
        upsampled_feature_5_to_4 = self.pyramid_transformation_512_to_256(self._upsample(pyramid_feature_5, pyramid_feature_4, scale_factor=2))   # deconv c5 to c4.size
        downsampled_feature_3_to_4 = self.pyramid_transformation_128_to_256(self.downsample_c3_to_c4(resnet_feature_3))                           # downsample c3 to c4 size
        downsampled_feature_2_to_4 = self.pyramid_transformation_64_to_256(self.downsample_c2_to_c4(resnet_feature_2))                            # downsample c2 to c4 size
        pyramid_feature_4 = torch.add(pyramid_feature_4, upsampled_feature_5_to_4)                                                                # add deconv_c5 to pyramid_c4
        pyramid_feature_4 = torch.add(pyramid_feature_4, downsampled_feature_3_to_4)                                                                  # add maxpool_c3 to pyramid_c4
        pyramid_feature_4 = torch.add(pyramid_feature_4, downsampled_feature_2_to_4)    
        pyramid_feature_4 = self.upsample_transform_4(pyramid_feature_4)                                                          # add maxpool_c3 to pyramid_c4
        
        # Calculate pyramid_feature_5
        downsampled_feature_4_to_5 = self.pyramid_transformation_256_to_512(self.downsample_c4_to_c5(resnet_feature_4))                           # downsample c4 to c5 size
        downsampled_feature_3_to_5 = self.pyramid_transformation_128_to_512(self.downsample_c3_to_c5(resnet_feature_3))                           # downsample c3 to c5 size
        downsampled_feature_2_to_5 = self.pyramid_transformation_64_to_512(self.downsample_c2_to_c5(resnet_feature_2))                            # downsample c2 to c5 size
        pyramid_feature_5 = torch.add(pyramid_feature_5, downsampled_feature_4_to_5)                                                              # add maxpool_c4 to pyramid_c5
        pyramid_feature_5 = torch.add(pyramid_feature_5, downsampled_feature_3_to_5)                                                              # add maxpool_c3 to pyramid_c5
        pyramid_feature_5 = torch.add(pyramid_feature_5, downsampled_feature_2_to_5)                                                              # add maxpool_c2 to pyramid_c5
        pyramid_feature_5 = self.upsample_transform_5(pyramid_feature_5)

        return [pyramid_feature_2, pyramid_feature_3, pyramid_feature_4, pyramid_feature_5]

class FPN3D(nn.Module):
    backbones = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }

    def __init__(self, cfg, backbone='resnet34', num_classes=1, pretrained=False):
        super(FPN3D, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.backbone_net = FPN3D.backbones[backbone](pretrained=pretrained)
        self.feature_pyramid = FeaturePyramid(self.backbone_net)
        self.output = nn.Sequential(nn.Conv3d(256, 64, kernel_size = 1),
                                    nn.ReLU(),
                                    nn.Dropout3d(p = 0.3),
                                    nn.Conv3d(64, 5 * len(config['anchors']), kernel_size = 1))

    def forward(self, x, coord):
        pyramid_features = self.feature_pyramid(x)[0]
        print(pyramid_features.shape)
        out = self.output(pyramid_features)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        return out

def get_model():
    cfg = {
        'in_planes': (24,48,72,96),
        'out_planes': (24,48,72,96),
        'num_blocks': (2,2,2,2), # block number in one blocks
        'dense_depth': (8,8,8,8)
    }
    net = FPN3D(cfg)
    print('Model----3dfpn!')
    loss = Loss(config['num_hard'])
    get_pbb = GetPBB(config) # TODO: Understand
    return config, net, loss, get_pbb

def test():
    debug = True
    _, net, _, _ = get_model()
    x = Variable(torch.randn(2,1,96,96,96), requires_grad=True)
    crd = Variable(torch.randn(2,3,24,24,24), requires_grad=True)
    y = net(x, crd)
    print(y)

test()