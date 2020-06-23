# -*- coding: utf-8 -*-
import os
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets.data_io import get_transform, read_all_lines
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

processed = transforms.Compose([
        transforms.ToTensor(),
    ]
)

def augment_image_pair(left_image, right_image):
    random_gamma = torch.rand(1).numpy()[0] * 0.3 + 0.6
    left_image_aug  = left_image  ** random_gamma
    right_image_aug = right_image ** random_gamma

    random_brightness = torch.rand(1).numpy()[0] * 1.5 + 0.5
    left_image_aug = left_image_aug * random_brightness
    right_image_aug = right_image_aug * random_brightness
    left_image_aug = torch.clamp(left_image_aug, 0, 1)
    right_image_aug = torch.clamp(right_image_aug, 0, 1)
    return left_image_aug, right_image_aug

def PngToTensor(image):
    img_tensor = processed(image).float()
    image_tensor = img_tensor / 65535.0
    return image_tensor

class data_load(Dataset):
    def __init__(self, datapath, list_filename, training, crop_h, crop_w, channels):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.channels = channels

    def load_path(self, list_filename):
        with open(list_filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_img_png(self, filename):
        return Image.open(filename)

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_img_png(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_img_png(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_img_png(os.path.join(self.datapath, self.disp_filenames[index]))

        if self.training:
            w, h = left_img.size
            x1 = random.randint(0, w - self.crop_w)
            y1 = random.randint(0, h - self.crop_h)
            left_img = left_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            right_img = right_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            disparity = disparity.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h)) 
            disparity = np.array(disparity)

            left_img = processed(left_img)
            right_img = PngToTensor(right_img)
            disparity = torch.from_numpy(disparity)

            do_augment = torch.rand(1).numpy()[0]
            if do_augment > 0.5:
                left_img, right_img = augment_image_pair(left_img, right_img)

            return{"left": left_img,
                   "right": right_img,
                   "disp_input": disparity,}
        else:
            w, h = left_img.size
            x1 = random.randint(0, w - self.crop_w)
            y1 = random.randint(0, h - self.crop_h)

            left_img = left_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            right_img = right_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            left_img = PngToTensor(left_img)
            right_img = PngToTensor(right_img)

            return {"left": left_img,
                    "right": right_img}
