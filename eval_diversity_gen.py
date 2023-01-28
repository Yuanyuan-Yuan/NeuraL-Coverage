import os
import sys
import copy
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import data_loader
import utility
import models
import tool
import coverage
import constants

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ImageNet', choices=['ImageNet', 'CIFAR10'])
parser.add_argument('--model', type=str, default='BigGAN', choices=['BigGAN'])
parser.add_argument('--criterion', type=str, default='NLC', 
                    choices=['NLC', 'NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'TKNP', 'CC',
                    'LSC', 'DSC', 'MDSC'])
parser.add_argument('--output_dir', type=str, default='./test_folder')
# parser.add_argument('--nc', type=int, default=3)
# parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_class', type=float, default=100)
parser.add_argument('--num_per_class', type=float, default=500)
parser.add_argument('--hyper', type=float, default=None)
args = parser.parse_args()
args.exp_name = ('%s-%s-%s-%s' % (args.dataset, args.model, args.criterion, args.hyper))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

utility.make_path(args.output_dir)
fp = open('%s/%s.txt' % (args.output_dir, args.exp_name), 'w')

utility.log(('Dataset: %s \nModel: %s \nCriterion: %s \nHyper-parameter: %s' % (
    args.dataset, args.model, args.criterion, args.hyper
)), fp)

USE_SC = args.criterion in ['LSC', 'DSC', 'MDSC']

if args.dataset == 'ImageNet':
    sys.path.insert(0, constants.BIGGAN_IMAGENET_PROJECT_DIR) # ImageNet
    Z_DIM = constants.BIGGAN_IMAGENET_LATENT_DIM
    TOTAL_CLASS_NUM = 1000
elif args.dataset == 'CIFAR10':
    sys.path.insert(0, constants.BIGGAN_CIFAR10_PROJECT_DIR) # ImageNet
    Z_DIM = constants.BIGGAN_CIFAR10_LATENT_DIM
    TOTAL_CLASS_NUM = 10

from for_import import get_G, gen_from_z_y
model = get_G(os.path.join(constants.PRETRAINED_MODELS, 'BigGAN/%s/' % args.dataset))

noise = torch.randn((args.batch_size, Z_DIM)).to(DEVICE)
label = torch.zeros(args.batch_size).type(torch.LongTensor).to(DEVICE)
random_data = (noise, model.shared(label))
layer_size_dict = tool.get_layer_output_sizes(model, random_data)

num_neuron = 0
for layer_name in layer_size_dict.keys():
    num_neuron += layer_size_dict[layer_name][0]
print('Total %d layers: ' % len(layer_size_dict.keys()))
print('Total %d neurons: ' % num_neuron)

if USE_SC:
    criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=args.hyper, min_var=1e-5, num_class=TOTAL_CLASS_NUM)
else:
    criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=args.hyper)

NUM_BATCH = 100

train_loader = []
print('Randomly sampling noise to simulate training inputs...')
for _ in tqdm(range(NUM_BATCH)):
    noise = torch.randn((args.batch_size, Z_DIM)).to(DEVICE)
    label = torch.randint(high=int(args.num_class), size=(args.batch_size,)).type(torch.LongTensor).to(DEVICE)
    data = (noise, model.shared(label))
    train_loader.append((data, label))
'''
KMNC/NBC/SNAC/LSC/DSC/MDSC requires training inputs, which are random noise
when training generative models.
'''

criterion.build(train_loader)
if args.criterion not in ['CC', 'TKNP', 'LSC', 'DSC', 'MDSC']:
    criterion.assess(train_loader)
'''
For LSC/DSC/MDSC/CC/TKNP, initialization with training data is too slow (sometimes may
exceed the memory limit). You can skip this step to speed up the experiment, which
will not affect the conclusion because we only compare the relative order of coverage
values, rather than the exact numbers.
'''
utility.log('Initial coverage: %d' % criterion.current, fp)

criterion1 = copy.deepcopy(criterion)
for _ in tqdm(range(NUM_BATCH)):
    noise = torch.randn((args.batch_size, Z_DIM)).to(DEVICE)
    label = torch.randint(high=int(args.num_class), size=(args.batch_size,)).type(torch.LongTensor).to(DEVICE)
    data = (noise, model.shared(label))
    if USE_SC:
        criterion1.step(data, label)
    else:
        criterion1.step(data)
utility.log(('Test: %f, increase: %f' % (criterion1.current, criterion1.current - criterion.current)), fp)
del criterion1

for divisor in [2, 10]:
    # 1 / divisor of inputs, all classes
    criterion2 = copy.deepcopy(criterion)
    for _ in tqdm(range(int(NUM_BATCH / divisor))):
        noise = torch.randn((args.batch_size, Z_DIM)).to(DEVICE)
        label = torch.randint(high=int(args.num_class), size=(args.batch_size,)).type(torch.LongTensor).to(DEVICE)
        data = (noise, model.shared(label))
        if USE_SC:
            criterion2.step(data, label)
        else:
            criterion2.step(data)
    utility.log(('N/%d: %f, increase: %f' % (divisor, criterion2.current, criterion2.current - criterion.current)), fp)
    del criterion2

    # 1 / divisor of inputs, 1 / divisor of classes
    criterion2 = copy.deepcopy(criterion)
    for _ in tqdm(range(int(NUM_BATCH / divisor))):
        noise = torch.randn((args.batch_size, Z_DIM)).to(DEVICE)
        label = torch.randint(high=int(args.num_class / divisor), size=(args.batch_size,)).type(torch.LongTensor).to(DEVICE)
        data = (noise, model.shared(label))
        if USE_SC:
            criterion2.step(data, label)
        else:
            criterion2.step(data)
    utility.log(('C/%d: %f, increase: %f' % (divisor, criterion2.current, criterion2.current - criterion.current)), fp)
    del criterion2