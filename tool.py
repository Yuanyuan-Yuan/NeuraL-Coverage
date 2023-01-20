import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import copy


def scale(out, dim=-1, rmax=1, rmin=0):
    out_max = out.max(dim)[0].unsqueeze(dim)
    out_min = out.min(dim)[0].unsqueeze(dim)
    '''
        out_max = out.max()
        out_min = out.min()
    Note that the above max/min is incorrect when batch_size > 1
    '''
    output_std = (out - out_min) / (out_max - out_min)
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled

def is_valid(module):
    return (isinstance(module, nn.Linear)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Conv1d)
            # or isinstance(module, nn.Conv3d)
            # or isinstance(module, nn.RNN)
            # or isinstance(module, nn.LSTM)
            # or isinstance(module, nn.GRU)
            )

def iterate_module(name, module, name_list, module_list):
    if is_valid(module):
        return name_list + [name], module_list + [module]
    else:

        if len(list(module.named_children())):
            for child_name, child_module in module.named_children():
                name_list, module_list = \
                    iterate_module(child_name, child_module, name_list, module_list)
        return name_list, module_list

'''
The implementation of Pytorch hook is based on
https://github.com/fabriceyhc/nc_diversity_attacks/blob/master/neuron_coverage.py
'''

def get_model_layers(model):
    layer_dict = {}
    name_counter = {}
    for name, module in model.named_children():
        name_list, module_list = iterate_module(name, module, [], [])
        assert len(name_list) == len(module_list)
        for i, _ in enumerate(name_list):
            module = module_list[i]
            class_name = module.__class__.__name__
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1
            layer_dict['%s-%d' % (class_name, name_counter[class_name])] = module
    # DEBUG
    # print('layer name')
    # for k in layer_dict.keys():
    #     print(k, ': ', layer_dict[k])
    
    return layer_dict

def get_layer_output_sizes(model, data, layer_name=None):   
    output_sizes = {}
    hooks = []  
    
    layer_dict = get_model_layers(model)
 
    def hook(module, input, output):
        module_idx = len(output_sizes)
        m_key = list(layer_dict)[module_idx]
        output_sizes[m_key] = list(output.size()[1:])
        # output_sizes[m_key] = list(output.size())
    
    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))
    
    try:
        if type(data) is tuple:
            model(*data)
        else:
            model(data)
    finally:
        for h in hooks:
            h.remove()
    # DEBUG
    # print('output size')
    # for k in output_sizes.keys():
    #     print(k, ': ', output_sizes[k])

    return output_sizes

def get_layer_output(model, data, threshold=0.5, force_relu=False, layer_name=None):
    with torch.no_grad():    
        layer_dict = get_model_layers(model)

        layer_output_dict = {}
        def hook(module, input, output):
            module_idx = len(layer_output_dict)
            m_key = list(layer_dict)[module_idx]
            if force_relu:
                output = F.relu(output)
            layer_output_dict[m_key] = output # (N, K, H, W) or (N, K)

        hooks = []
        for layer, module in layer_dict.items():
            hooks.append(module.register_forward_hook(hook))
        try:
            if type(data) is tuple:
                final_out = model(*data)
            else:
                final_out = model(data)

        finally:
            for h in hooks:
                h.remove()

        for layer, output in layer_output_dict.items():
            assert len(output.size()) == 2 or len(output.size()) == 4
            if len(output.size()) == 4: # (N, K, H, w)
                output = output.mean((2, 3))
            layer_output_dict[layer] = output.detach()
        return layer_output_dict


class Estimator(object):
    def __init__(self, feature_num, class_num=1):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).to(self.device)
        self.Ave = torch.zeros(class_num, feature_num).to(self.device)
        self.Amount = torch.zeros(class_num).to(self.device)

    def calculate(self, features, labels=None):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        if labels is None:
            labels = torch.zeros(N).to(self.device)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).to(self.device)
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        # self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
        #               .mul(weight_CV)).detach() + additional_CV.detach()

        # self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        # self.Amount += onehot.sum(0)

        new_CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                      .mul(weight_CV)).detach() + additional_CV.detach()

        new_Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        new_Amount = self.Amount + onehot.sum(0)

        return {
            'Ave': new_Ave, 
            'CoVariance': new_CoVariance,
            'Amount': new_Amount
        }

    def update(self, dic):
        self.Ave = dic['Ave']
        self.CoVariance = dic['CoVariance']
        self.Amount = dic['Amount']

    def transform(self, features, labels):
        CV = self.CoVariance[labels]
        (N, A) = features.size()
        transformed = torch.bmm(F.normalize(CV), features.view(N, A, 1))
        return transformed.squeeze(-1)

class EstimatorFlatten(object):
    def __init__(self, feature_num, class_num=1):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num).to(self.device)
        self.Ave = torch.zeros(class_num, feature_num).to(self.device)
        self.Amount = torch.zeros(class_num).to(self.device)

    def calculate(self, features, labels=None):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        if labels is None:
            labels = torch.zeros(N).to(self.device)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).to(self.device)
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        
        new_CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        new_Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        new_Amount = self.Amount + onehot.sum(0)

        return {
            'Ave': new_Ave,
            'CoVariance': new_CoVariance,
            'Amount': new_Amount
        }

    def update(self, dic):
        self.Ave = dic['Ave']
        self.CoVariance = dic['CoVariance']
        self.Amount = dic['Amount']

    def transform(self, features, labels):
        CV = self.CoVariance[labels]
        (N, A) = features.size()
        transformed = torch.bmm(features.view(N, 1, A), F.normalize(CV))
        return transformed.transpose(1, 2).squeeze(-1)


if __name__ == '__main__':
    pass