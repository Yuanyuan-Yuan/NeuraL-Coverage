#!/usr/bin/env python
import os
import argparse
from pathlib import Path
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms

import constants

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.data.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.data.size()[:2] == style_feat.data.size()[:2])
    size = content_feat.data.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

# random.seed(131213)

def input_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(torchvision.transforms.Resize(size))
    if crop != 0:
        transform_list.append(torchvision.transforms.CenterCrop(crop))
    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)
    return transform

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    with torch.no_grad():    
        assert (0.0 <= alpha <= 1.0)
        content_f = vgg(content)
        style_f = vgg(style)
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return decoder(feat)

class Stylized(object):
    def __init__(self, style_size):
        self.init_params(style_size)
        self.init_models()
        self.prepare_style()
        self.content_tf = input_transform(self.content_size, self.crop)
        self.style_tf = input_transform(self.style_size, 0)

    def init_params(self, style_size):
        self.style_dir = constants.STYLE_IMAGE_DIR
        self.num_styles = 1
        # self.alpha = 1.0
        self.extensions = ['png', 'jpeg', 'jpg']
        self.content_size = 0
        self.style_size = style_size
        self.crop = 0
        self.decoder_path = os.path.join(constants.STYLE_MODEL_DIR, 'decoder.pth')
        self.vgg_path = os.path.join(constants.STYLE_MODEL_DIR, 'vgg_normalised.pth')

    def prepare_style(self):
        style_dir = Path(self.style_dir)
        style_dir = style_dir.resolve()
        assert style_dir.is_dir(), 'Style directory not found'
        
        styles = []
        for ext in self.extensions:
            styles += list(style_dir.rglob('*.' + ext))

        assert len(styles) > 0, 'No images with specified extensions found in style directory' + self.style_dir
        self.styles = sorted(styles)
        print('Found %d style images in %s' % (len(self.styles), self.style_dir))

    def init_models(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = decoder
        self.vgg = vgg
        self.decoder.eval()
        self.vgg.eval()

        self.decoder.load_state_dict(torch.load(self.decoder_path))
        self.vgg.load_state_dict(torch.load(self.vgg_path))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        self.vgg.to(self.device)
        self.decoder.to(self.device)

    def transform(self, image, alpha):
        if type(image) is str:
            content_image = Image.open(image).convert('RGB')
            content_image = self.content_tf(content_image).to(self.device)
        elif type(image) is np.ndarray:
            content_image = torch.from_numpy(image / 255).to(self.device)
            if len(content_image.size()) == 3:
                content_image = content_image.transpose(0, 2).unsqueeze(0)
            elif len(content_image.size()) == 4:
                content_image = content_image.transpose(1, 3)
            else:
                raise NotImplementedError
        else:
            content_image = image

        style_path = random.sample(self.styles, self.num_styles)[0]
        style_image = Image.open(style_path).convert('RGB')
        style_image = self.style_tf(style_image)
        style_image = style_image.to(self.device).unsqueeze(0)

        output = style_transfer(self.vgg, self.decoder,
                                content_image, style_image, alpha)

        if type(image) is np.ndarray:
            output = output * 255
            if len(content_image.size()) == 3:
                output = output.transpose(0, 2).detach().cpu().numpy()
            elif len(content_image.size()) == 4:
                output = output.transpose(1, 3).detach().cpu().numpy()
            else:
                raise NotImplementedError
        else:
            output = output.detach().cpu()
        
        return output
