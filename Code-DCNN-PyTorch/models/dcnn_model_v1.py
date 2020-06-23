from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import math
from .basic_module import *

# feature extraction
class feature_extraction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(feature_extraction, self).__init__()
        # conv0_x
        self.conv0_x = nn.Sequential(
            # conv0_1                                                  
            nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # conv0_2
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),        
        )

        # conv2_x
        self.conv2_x = nn.Sequential(
            convbn(32, 64, 3, 2, 1, 1), 
            nn.ReLU(inplace=True),         
            BasicBlock(64, 64, 1, 1, 1),
            BasicBlock(64, 64, 1, 1, 1),  
        )

        # conv3_x
        self.conv3_x = nn.Sequential(
            convbn(64, 128, 1, 1, 0, 1),  
            nn.ReLU(inplace=True),
            BasicBlock(128, 128, 1, 1, 1),
            BasicBlock(128, 128, 1, 1, 1),
            BasicBlock(128, 128, 1, 1, 1),
            convbn(128, 128, 3, 1, 1, 2),
            nn.ReLU(inplace=True),
        )

        self.lastconv_1 = nn.Sequential(convbn(192, 128, 3, 1, 1, 1),  
                                        nn.ReLU(inplace=True),
                                        convbn(128, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True))  
        self.lastconv_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, bias=False))

    def forward(self, x):
        out0_x = self.conv0_x(x)
        out2_x = self.conv2_x(out0_x)
        out3_x = self.conv3_x(out2_x) 
        concat_feature_192 = torch.cat((out2_x, out3_x), 1)  # 192 * H/4 * W/4
        # 3*3 + 1*1
        out = self.lastconv_1(concat_feature_192)  # 32 * H/4 * W/4
        out_feature = self.lastconv_2(out) 
        return out_feature

# model
class dcnn_model(nn.Module):
    def __init__(self, in_channel, out_channel, max_disp):
        super(dcnn_model, self).__init__()
        self.max_disp = max_disp
        self.feature_extraction = feature_extraction(in_channel, out_channel)

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        l_f = self.feature_extraction(left)
        r_f = self.feature_extraction(right)
        cost_volume = Build_Concat_Volume(l_f, r_f, int(self.max_disp / 4))

        cost0 = self.dres0(cost_volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.upsample(cost0, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.max_disp)

            cost1 = F.upsample(cost1, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.max_disp)

            cost2 = F.upsample(cost2, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.max_disp)

            cost3 = F.upsample(cost3, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.max_disp)
            return [pred0, pred1, pred2, pred3]

        else:
            cost3 = self.classif3(out3)
            cost3 = F.upsample(cost3, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.max_disp)
            return [pred3]