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
    def __init__(self, down_num, in_channel, out_channel):
        super(feature_extraction, self).__init__()
        # 3/4-downsampling
        self.down1_x = nn.Conv2d(in_channel, out_channel, 5, stride=2, padding=2, bias=False)
        self.down2_x = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2, bias=False)
        if down_num == 3:
            self.down3_x = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2, bias=False),)
        else:
            self.down3_x = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2, bias=False),
                nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2, bias=False),)
        # 6-basicblock
        self.conv2_x = nn.Sequential(
            BasicBlock(32, 1, 1, 1),
            BasicBlock(32, 1, 1, 1),
            BasicBlock(32, 1, 1, 1),
            BasicBlock(32, 1, 1, 1),
            BasicBlock(32, 1, 1, 1),
            BasicBlock(32, 1, 1, 1),
        )
        # alone-nobn/relu
        self.conv_alone = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, x):
        outdown1_x = self.down1_x(x)
        outdown2_x = self.down2_x(outdown1_x)
        outdown3_x = self.down3_x(outdown2_x)
        out2_x = self.conv2_x(outdown3_x)
        out_feature = self.conv_alone(out2_x)
        return out_feature

# Hierarchical refinement
class edge_refinement(nn.Module):
    def __init__(self, in_channel):
        super(edge_refinement, self).__init__()
        self.convbn_alone = nn.Sequential(
            convbn(in_channel, 32, 3, 1, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_atrous = nn.Sequential(
            BasicBlock(32, 32, 1, 1, 1),
            BasicBlock(32, 32, 1, 1, 2),
            BasicBlock(32, 32, 1, 1, 4),
            BasicBlock(32, 32, 1, 1, 8),
            BasicBlock(32, 32, 1, 1, 1),
            BasicBlock(32, 32, 1, 1, 1),
        )
        self.conv_alone = nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
    def forward(self, x):
        out_convbn = self.convbn_alone(x)
        out_atrous = self.conv_atrous(out_convbn)
        out_convalone = self.conv_alone(out_atrous)
        return out_convalone

# model
class stereo_model(nn.Module):
    def __init__(self, down_num, in_channel, out_channel, max_disp, use_edge_refinement=False):
        super(stereo_model, self).__init__()
        self.down_num = down_num
        self.max_disp = max_disp
        self.feature_extraction = feature_extraction(down_num, in_channel, out_channel)
        self.use_edge_refinemant = use_edge_refinement
        self.edge_refine = edge_refinement(4)

        self.conv3d = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3d_alone = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)

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
        # feature extraction
        l_f = self.feature_extraction(left)
        r_f = self.feature_extraction(right)
        # build_cost_volume(concat)
        down = 2**self.down_num
        cost_volume = Cost_Volume(l_f, r_f, int(self.madisp / down))
        # conv3d
        out_conv3d = self.conv3d(cost_volume)
        out_conv3d_alone = self.conv3d_alone(out_conv3d)
        # softmax, disp_regression
        out = F.upsample(out_conv3d_alone, [self.max_disp, left.size()[2], left.size()[3]], mode='trilinear')
        out = torch.squeeze(out, 1)
        pred = F.softmax(out, dim=1)
        pred = disparity_regression(pred, self.max_disp)

        if self.use_edge_refinement:
            pred_disp = torch.unsqueeze(pred, dim=1)
            pred_conc_left = torch.cat([pred_disp, left], 1)
            pred = self.edge_refine(pred_conc_left)
            return [pred]
        else:
            return [pred]
