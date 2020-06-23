# -*- coding: utf-8 -*-
import torch.nn.functional as F
import torch

#
def multi_model_loss(disp_ests, disp_gt, mask):
    weights = [0.1, 0.3, 0.4, 1.0]  # [0.5, 0.5, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)