import os
import json
import time
import random
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms


def log(content, f):
    print(content)
    print(content, file=f)

def image_normalize(image, dataset):
    if dataset == 'CIFAR10':
       transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    elif dataset == 'ImageNet':
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        raise NotImplementedError
    return transform(image)

def image_normalize_inv(image, dataset):
    if dataset == 'CIFAR10':
        transform = NormalizeInverse((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    elif dataset == 'ImageNet':
        transform = NormalizeInverse((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        raise NotImplementedError
    return transform(image)

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        # std_inv = 1 / (std + 1e-7)
        std_inv = 1 / std
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
        #return super().__call__(tensor)

class Logger(object):
    def __init__(self, args, engine):
        import time
        self.name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.log'
        self.args = args
        self.log_path = os.path.join(args.log_dir, self.name)
        self.f = open(self.log_path, 'a')
        self.f.write('Dataset: %s\n' % args.dataset)
        self.f.write('Model: %s\n' % args.model)
        self.f.write('Class: %d\n' % args.num_class)
        self.f.write('Data in each class: %d\n' % args.num_per_class)
        self.f.write('Criterion: %s\n' % args.criterion)

        for k in engine.hyper_params.keys():
            self.f.write('%s %s\n' % (k, engine.hyper_params[k]))
    
    def update(self, engine):
        print('Epoch: %d' % engine.epoch)
        print('Delta coverage: %f' % (engine.criterion.current - engine.initial_coverage))
        print('Delta time: %fs' % engine.delta_time)
        print('Delta batch: %d' % engine.delta_batch)
        print('AE: %d' % engine.num_ae)
        self.f.write('Delta time: %fs, Epoch: %d, Current coverage: %f, Delta coverage:%f, AE: %d, Delta batch: %d\n' % \
            (engine.delta_time, engine.epoch, engine.criterion.current, \
             engine.criterion.current - engine.initial_coverage,
             engine.num_ae, engine.delta_batch))

    def exit(self):
        self.f.close()