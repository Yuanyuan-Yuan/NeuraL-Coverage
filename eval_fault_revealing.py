import os
import copy
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import torchvision

import data_loader
import utility
import models
import tool
import coverage
import constants

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['ImageNet', 'CIFAR10'])
parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'vgg16_bn', 'mobilenet_v2'])
parser.add_argument('--criterion', type=str, default='NC', 
                    choices=['NLC', 'NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'TKNP', 'CC',
                    'LSC', 'DSC', 'MDSC'])
parser.add_argument('--AE', type=str, default='PGD', choices=['PGD', 'CW'])
parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])

parser.add_argument('--output_dir', type=str, default='./test_folder')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_class', type=float, default=10)
parser.add_argument('--num_per_class', type=float, default=5000)
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
    model = torchvision.models.__dict__[args.model](pretrained=False)
    path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pth' % (args.dataset, args.model)))
    assert args.image_size == 128
    assert args.num_class <= 1000
elif args.dataset == 'CIFAR10':
    model = getattr(models, args.model)(pretrained=False)
    path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pt' % (args.dataset, args.model)))
    assert args.image_size == 32
    assert args.num_class <= 10

TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = data_loader.get_loader(args)

ae_file = '%s-%s-%s-%s.pt' % (args.AE, args.model, args.dataset, args.split)
adv_images, labels = torch.load(os.path.join(constants.AE_DIR, ae_file))
adv_data = TensorDataset(adv_images, labels)
ae_loader = torch.utils.data.DataLoader(
        adv_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

model.load_state_dict(torch.load(path))
model.to(DEVICE)
model.eval()

input_size = (1, args.nc, args.image_size, args.image_size)
random_data = torch.randn(input_size).to(DEVICE)
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
criterion1.assess(test_loader)
utility.log(('Test: %f, increase: %f' % (criterion1.current, criterion1.current - criterion.current)), fp)
del criterion1

criterion2 = copy.deepcopy(criterion)
criterion2.assess(ae_loader)
utility.log(('%s: %f, increase: %f' % (args.AE, criterion2.current, criterion2.current - criterion.current)), fp)
del criterion2

