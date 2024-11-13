import sys
import copy
import random
import numpy as np
import time
from tqdm import tqdm
import itertools
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.utils import save_image

import coverage
import utility
from style_operator import Stylized
import image_transforms

class Parameters(object):
    def __init__(self, base_args):
        self.model = base_args.model
        self.dataset = base_args.dataset
        self.criterion = base_args.criterion
        self.use_sc = self.criterion in ['LSC', 'DSC', 'MDSC']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4
        
        self.batch_size = 50
        self.mutate_batch_size = 1
        self.nc = 3
        self.image_size = 128 if self.dataset == 'ImageNet' else 32
        self.input_shape = (1, self.image_size, self.image_size, 3)
        self.num_class = 100 if self.dataset == 'ImageNet' else 10
        self.num_per_class = 1000 // self.num_class

        self.input_scale = 255
        self.noise_data = False
        self.K = 64
        self.batch1 = 64
        self.batch2 = 16

        self.alpha = 0.2 # default 0.02
        self.beta = 0.5 # default 0.2
        self.TRY_NUM = 50
        self.save_every = 100
        self.output_dir = '/data/yyuanaq/output/Coverage/Fuzzer/'

        translation = list(itertools.product([getattr(image_transforms, "image_translation")],
                                            [(-5, -5), (-5, 0), (0, -5), (0, 0), (5, 0), (0, 5), (5, 5)]))        
        scale = list(itertools.product([getattr(image_transforms, "image_scale")], list(np.arange(0.8, 1, 0.05))))
        # shear = list(itertools.product([getattr(image_transforms, "image_shear")], list(range(-3, 3))))
        rotation = list(itertools.product([getattr(image_transforms, "image_rotation")], list(range(-30, 30))))

        contrast = list(itertools.product([getattr(image_transforms, "image_contrast")], [0.8 + 0.2 * k for k in range(7)]))
        brightness = list(itertools.product([getattr(image_transforms, "image_brightness")], [10 + 10 * k for k in range(7)]))
        blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k + 1 for k in range(10)]))

        self.stylized = Stylized(self.image_size)

        self.G = translation + scale + rotation #+ shear
        self.P = contrast + brightness + blur
        self.S = list(itertools.product([self.stylized.transform], [0.4, 0.6, 0.8]))
        self.save_batch = False

class INFO(object):
    def __init__(self):
        self.dict = {}

    def __getitem__(self, i):
        _i = str(i)
        if _i in self.dict:
            return self.dict[_i]
        else:
            I0, state = i, 0
            return I0, state

    def __setitem__(self, i, s):
        _i = str(i)
        self.dict[_i] = s
        return self.dict[_i]

class Fuzzer:
    def __init__(self, params, criterion):
        self.params = params
        self.epoch = 0
        self.time_slot = 60 * 10
        self.time_idx = 0
        self.info = INFO()
        self.hyper_params = {
            'alpha': 0.4, # [0, 1], default 0.02, 0.1 # number of pix
            'beta': 0.8, # [0, 1], default 0.2, 0.5 # max abs pix
            'TRY_NUM': 50,
            'p_min': 0.01,
            'gamma': 5,
            'K': 64
        }
        self.logger = utility.Logger(params, self)
        self.criterion = criterion
        self.initial_coverage = copy.deepcopy(criterion.current)
        self.delta_time = 0
        self.delta_batch = 0
        self.num_ae = 0

    def exit(self):
        self.print_info()
        self.criterion.save(self.params.coverage_dir + 'coverage.pt')
        self.logger.exit()

    def can_terminate(self):
        condition = sum([
            self.epoch > 10000,
            self.delta_time > 60 * 60 * 6,
        ]) 
        return condition > 0

    def print_info(self):
        self.logger.update(self)

    def is_adversarial(self, image, label, k=1):
        with torch.no_grad():
            scores = self.criterion.model(image)
            _, ind = scores.topk(k, dim=1, largest=True, sorted=True)
            correct = ind.eq(label.view(-1, 1).expand_as(ind))
            wrong = ~correct
            index = (wrong == True).nonzero(as_tuple=True)[0]
            wrong_total = wrong.view(-1).float().sum()
            return wrong_total, index

    def to_batch(self, data_list):
        batch_list = []
        batch = []
        for i, data in enumerate(data_list):
            if i and i % self.params.mutate_batch_size == 0:
                batch_list.append(np.stack(batch, 0))
                batch = []
            batch.append(data)
        if len(batch):
            batch_list.append(np.stack(batch, 0))
        return batch_list

    def image_to_input(self, image):
        scaled_image = image / self.params.input_scale
        tensor_image = torch.from_numpy(scaled_image).transpose(1, 3)
        normalized_image = utility.image_normalize(tensor_image, self.params.dataset)
        return normalized_image

    def run(self, I_input, L_input):
        F = np.array([]).reshape(0, *(self.params.input_shape[1:])).astype('float32')
        T = self.Preprocess(I_input, L_input)

        del I_input
        del L_input
        gc.collect()

        B, B_label, B_id = self.SelectNext(T)
        self.epoch = 0
        start_time = time.time()
        while not self.can_terminate():
            if self.epoch % 500 == 0:
                self.print_info()
            # S = self.Sample(B)
            S = B
            S_label = B_label
            Ps = self.PowerSchedule(S, self.hyper_params['K'])
            B_new = np.array([]).reshape(0, *(self.params.input_shape[1:])).astype('float32')
            B_old = np.array([]).reshape(0, *(self.params.input_shape[1:])).astype('float32')
            B_label_new = []
            for s_i in range(len(S)):
                I = S[s_i]
                L = S_label[s_i]
                I_new, op = self.Mutate(I)

                if self.CoverageGain():
                    # always returns True
                    B_new = np.concatenate((B_new, [I_new]))
                    B_old = np.concatenate((B_old, [I]))
                    B_label_new += [L]

            if len(B_new) > 0:
                B_label_new = np.array(B_label_new)
                new_image = self.image_to_input(B_new)
                new_image = new_image.to(self.params.device)
                new_label = torch.from_numpy(B_label_new)
                new_label = new_label.to(self.params.device)
                
                B_c, Bs, Bs_label = T
                B_c += [0]
                Bs += [B_new]
                Bs_label += [B_label_new]
                self.delta_batch += 1
                self.BatchPrioritize(T, B_id)

                num_wrong, ae_index = self.is_adversarial(new_image, new_label)
                if num_wrong > 0:
                    self.num_ae += num_wrong

                if self.epoch % self.params.save_every == 0:
                    self.saveImage(B_new / self.params.input_scale, self.params.image_dir + ('%03d_new.jpg' % self.epoch))
                    self.saveImage(B_old / self.params.input_scale, self.params.image_dir + ('%03d_old.jpg' % self.epoch))
                    if num_wrong > 0:
                        print('Saving AE images...')
                        save_image(new_image[ae_index].data, self.params.image_dir + ('%03d_ae.jpg' % self.epoch), normalize=True)

            gc.collect()

            B, B_label, B_id = self.SelectNext(T)
            self.epoch += 1
            self.delta_time = time.time() - start_time


    def Preprocess(self, image_list, label_list):        
        randomize_idx = np.arange(len(image_list))
        np.random.shuffle(randomize_idx)
        image_list = [image_list[idx] * self.params.input_scale for idx in randomize_idx]
        label_list = [label_list[idx] for idx in randomize_idx]

        Bs = self.to_batch(image_list)
        Bs_label = self.to_batch(label_list)

        return list(np.zeros(len(Bs))), Bs, Bs_label

    def calc_priority(self, B_ci):
        if B_ci < (1 - self.hyper_params['p_min']) * self.hyper_params['gamma']:
            return 1 - B_ci / self.hyper_params['gamma']
        else:
            return self.hyper_params['p_min']

    def SelectNext(self, T):
        B_c, Bs, Bs_label = T
        B_p = [self.calc_priority(B_c[i]) for i in range(len(B_c))]
        c = np.random.choice(len(Bs), p=B_p / np.sum(B_p))
        return Bs[c], Bs_label[c], c

    def Sample(self, B):
        c = np.random.choice(len(B), size=self.params.mutate_batch_size, replace=False)
        return B[c]

    def PowerSchedule(self, S, K):
        potentials = []
        for i in range(len(S)):
            I = S[i]
            I0, state = self.info[I]
            p = self.hyper_params['beta'] * 255 * np.sum(I > 0) - np.sum(np.abs(I - I0))
            potentials.append(p)
        potentials = np.array(potentials) / np.sum(potentials)

        def Ps(I_id):
            p = potentials[I_id]
            return int(np.ceil(p * K))
        return Ps

    def isFailedTest(self, I_new):
        return False

    def isChanged(self, I, I_new):
        return np.any(I != I_new)

    def CoverageGain(self):
        return True

    def BatchPrioritize(self, T, B_id):
        B_c, Bs, Bs_label = T
        B_c[B_id] += 1

    def Mutate(self, I):
        G, P, S = self.params.G, self.params.P, self.params.S
        I0, state = self.info[I]

        for i in range(1, self.hyper_params['TRY_NUM']):
            if state == 0:
                t, p = self.randomPick(G + P + S)
            else:
                t, p = self.randomPick(P + S)

            I_mutated = t(I, p).reshape(*(self.params.input_shape[1:]))
            I_mutated = np.clip(I_mutated, 0, 255)

            if (t, p) in S or self.f(I0, I_mutated):
                if (t, p) in G:
                    state = 1
                    I0_G = t(I0, p)
                    I0_G = np.clip(I0_G, 0, 255)
                    self.info[I_mutated] = (I0_G, state)
                else:
                    self.info[I_mutated] = (I0, state)
                return I_mutated, (t, p)
        return I, (t, p)

    def saveImage(self, image, path):
        if image is not None:
            print('Saving mutated images in %s...' % path)
            image_tensor = torch.from_numpy(image).transpose(1, 3)
            save_image(image_tensor.data, path, normalize=True)

    def randomPick(self, A):
        c = np.random.randint(0, len(A))
        return A[c]


    def f(self, I, I_new):
        if (np.sum((I - I_new) != 0) < self.hyper_params['alpha'] * np.sum(I > 0)):
            return np.max(np.abs(I - I_new)) <= 255
        else:
            return np.max(np.abs(I - I_new)) <= self.hyper_params['beta'] * 255


if __name__ == '__main__':
    import os
    import argparse
    import torchvision
    import gc

    import utility
    import models
    import tool
    import coverage
    import constants
    import data_loader

    import signal
    def signal_handler(sig, frame):
            print('You pressed Ctrl+C!')
            try:
                if engine is not None:
                    engine.print_info()
                    if engine.logger is not None:
                        engine.logger.exit()
                    if engine.criterion is not None:
                        engine.criterion.save(args.coverage_dir + 'coverage_int.pth')
            except:
                pass
            sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    os.environ['TORCH_HOME'] = '/data/yyuanaq/collection/ImageNet/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                            choices=['CIFAR10', 'ImageNet'])
    parser.add_argument('--model', type=str, default='resnet50',
                            choices=['resnet50', 'vgg16_bn', 'mobilenet_v2'])
    parser.add_argument('--criterion', type=str, default='NC', 
                            choices=['NLC', 'NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'TKNP', 'CC',
                    'LSC', 'DSC', 'MDSC'])
    parser.add_argument('--output_dir', type=str, default='./test_folder')
    # parser.add_argument('--hyper', type=str, default=None)
    base_args = parser.parse_args()

    args = Parameters(base_args)

    args.exp_name = ('rand-%s-%s-%s' % (args.dataset, args.model, args.criterion))
    print(args.exp_name)
    utility.make_path(args.output_dir)
    utility.make_path(args.output_dir + args.exp_name)

    args.image_dir = args.output_dir + args.exp_name + '/image/'
    args.coverage_dir = args.output_dir + args.exp_name + '/coverage/'
    args.log_dir = args.output_dir + args.exp_name + '/log/'

    utility.make_path(args.image_dir)
    utility.make_path(args.coverage_dir)
    utility.make_path(args.log_dir)

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

    model.load_state_dict(torch.load(path))
    model.to(args.device)
    model.eval()

    input_size = (1, args.nc, args.image_size, args.image_size)
    random_data = torch.randn(input_size).to(args.device)
    layer_size_dict = tool.get_layer_output_sizes(model, random_data)


    if args.dataset == 'CIFAR10':
        data_set = data_loader.CIFAR10FuzzDataset(args, split='test')
    elif args.dataset == 'ImageNet':
        data_set = data_loader.ImageNetFuzzDataset(args, split='val')

    TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = data_loader.get_loader(args)

    image_list, label_list = data_set.build()
    image_numpy_list = data_set.to_numpy(image_list)
    label_numpy_list = data_set.to_numpy(label_list, False)

    del image_list
    del label_list
    gc.collect()

    hyper_map = {
        'NLC': None,
        'NC': 0.75,
        'KMNC': 100,
        'SNAC': None,
        'NBC': None,
        'TKNC': 10,
        'TKNP': 50,
        'CC': 10 if args.dataset == 'CIFAR10' else 1000,
        'LSA': 10,
        'DSA': 0.1,
        'MDSA': 10
    }

    if args.use_sc:
        criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=hyper_map[args.criterion], min_var=1e-5, num_class=TOTAL_CLASS_NUM)
    else:
        criterion = getattr(coverage, args.criterion)(model, layer_size_dict, hyper=hyper_map[args.criterion])

    criterion.build(train_loader)
    if args.criterion not in ['CC', 'TKNP', 'LSC', 'DSC', 'MDSC']:
        criterion.assess(train_loader)
    '''
    For LSC/DSC/MDSC/CC/TKNP, initialization with training data is too slow (sometimes may
    exceed the memory limit). You can skip this step to speed up the experiment, which
    will not affect the conclusion because we only compare the relative order of coverage
    values, rather than the exact numbers.
    '''

    initial_coverage = copy.deepcopy(criterion.current)
    print('Initial Coverage: %f' % initial_coverage)
    engine = Fuzzer(args, criterion)
    engine.run(image_numpy_list, label_numpy_list)
    engine.exit()

