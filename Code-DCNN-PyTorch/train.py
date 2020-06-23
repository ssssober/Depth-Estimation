# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from models import multi_model_loss
from utils import *
from torch.utils.data import DataLoader
import gc

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='dcnn')
parser.add_argument('--mode', type=str, default='test', help='train or test')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='', help='data path')
parser.add_argument('--channels', type=int, default=3, help='net input channels')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--train_crop_height', type=int, default=256, help='training crop height')
parser.add_argument('--train_crop_width', type=int, default=256, help='training crop width')
parser.add_argument('--lr', type=float, default=0.0001, help='base learning rate')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')
parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--model', default='dcnn', help='select a model structure', choices=__models__.keys())
# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True, args.train_crop_height, args.train_crop_width, args.channels)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=6, drop_last=True)

train_file = open(args.trainlist, "r")
train_file_lines = train_file.readlines()
print("train_file_lines nums: ", len(train_file_lines))

# model, optimizer
model = __models__[args.model](args.channels, 32, args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

# output model parameters
print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))

# load parameters
start_epoch = 0
if args.resume:
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".tar")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    # model.load_state_dict(state_dict['model'])
    model.load_state_dict(state_dict['state_dict'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))

def train():
    train_start_time = time.time()
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()

            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                print('Epoch {}/{}, Iter {}/{}, Global_step {}/{}, train loss = {:.3f}, time = {:.3f}, time elapsed {:.3f}, time left {:.3f}h'.format(epoch_idx, args.epochs,
                                                                                           batch_idx, len(TrainImgLoader),
                                                                                           global_step, len(TrainImgLoader) * args.epochs,
                                                                                           loss,
                                                                                           time.time() - start_time,
                                                                                           (time.time() - train_start_time) / 3600,
                                                                                           (len(TrainImgLoader) * args.epochs / (global_step + 1) - 1) * (time.time() - train_start_time) / 3600))
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            # saving checkpoints
            if int(global_step / args.save_freq) != 0 and global_step % args.save_freq == 0:
                checkpoint_data = {'epoch': epoch_idx, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/checkpoint_{}_{:0>7}.tar".format(args.logdir, epoch_idx + 1, global_step))
        gc.collect()

# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()
    # training data load
    left_input, right_input, disp_gt= sample['left'], sample['right'], sample['disp_input']
    l = l.cuda()
    r = r.cuda()
    left_input = left_input.cuda()
    right_input = right_input.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()
    disp_ests = model(left_input, right_input)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    mask_gt = mask_gt.cuda()
    disp_gt = disp_gt.float()
    loss = multi_model_loss(disp_ests, disp_gt, mask_gt)
    mask_gt_gt = mask_gt.float()  #
    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "left": left_input, "right": right_input}
    # dict update
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask_gt) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask_gt) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask_gt, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask_gt, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask_gt, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

if __name__ == '__main__':
    if args.mode == 'train':
        train()
