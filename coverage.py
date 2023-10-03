from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyflann import FLANN
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

import tool


class Coverage:
    def __init__(self, model, layer_size_dict, hyper=None, **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.layer_size_dict = layer_size_dict
        self.init_variable(hyper, **kwargs)

    def init_variable(self):
        raise NotImplementedError
        
    def calculate(self):
        raise NotImplementedError

    def coverage(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def build(self, data_loader):
        print('Building is not needed.')

    def assess(self, data_loader):
        for data, *_ in tqdm(data_loader):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.step(data)

    def step(self, data):
        cove_dict = self.calculate(data)
        gain = self.gain(cove_dict)
        if gain is not None:
            self.update(cove_dict, gain)

    def update(self, all_cove_dict, delta=None):
        self.coverage_dict = all_cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(all_cove_dict)

    def gain(self, cove_dict_new):
        new_rate = self.coverage(cove_dict_new)
        return new_rate - self.current


class SurpriseCoverage(Coverage):
    def init_variable(self, hyper, min_var, num_class):
        self.name = self.get_name()
        assert self.name in ['LSC', 'DSC', 'MDSC']
        assert hyper is not None
        self.threshold = hyper
        self.min_var = min_var
        self.num_class = num_class
        self.data_count = 0
        self.current = 0
        self.coverage_set = set()
        self.mask_index_dict = {}
        self.mean_dict = {}
        self.var_dict = {}
        self.kde_cache = {}
        self.SA_cache = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.mask_index_dict[layer_name] = torch.ones(layer_size[0]).type(torch.LongTensor).to(self.device)
            self.mean_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)
            self.var_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)

    def get_name(self):
        raise NotImplementedError

    def build(self, data_loader):
        print('Building Mean & Var...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            # print(data.size())
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.set_meam_var(data, label)
        self.set_mask()
        print('Building SA...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            self.build_SA(data, label)
        self.to_numpy()
        if self.name == 'LSC':
            self.set_kde()
        if self.name == 'MDSC':
            self.compute_covinv()

    def assess(self, data_loader):
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            self.step(data, label)

    def step(self, data, label):
        cove_set = self.calculate(data, label)
        gain = self.gain(cove_set)
        if gain is not None:
            self.update(cove_set, gain)

    def set_meam_var(self, data, label):
        batch_size = label.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            self.data_count += batch_size
            self.mean_dict[layer_name] = ((self.data_count - batch_size) * self.mean_dict[layer_name] + layer_output.sum(0)) / self.data_count
            self.var_dict[layer_name] = (self.data_count - batch_size) * self.var_dict[layer_name] / self.data_count \
            + (self.data_count - batch_size) * ((layer_output - self.mean_dict[layer_name]) ** 2).sum(0) / self.data_count ** 2

    def set_mask(self):
        feature_num = 0
        for layer_name in self.mean_dict.keys():
            self.mask_index_dict[layer_name] = (self.var_dict[layer_name] >= self.min_var).nonzero()
            feature_num += self.mask_index_dict[layer_name].size(0)
        print('feature_num: ', feature_num)

    def build_SA(self, data_batch, label_batch):
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1) # [batch_size, num_neuron]
        # print('SA_batch: ', SA_batch.size())
        SA_batch = SA_batch[~torch.any(SA_batch.isnan(), dim=1)]
        SA_batch = SA_batch[~torch.any(SA_batch.isinf(), dim=1)]
        for i, label in enumerate(label_batch):
            if int(label.cpu()) in self.SA_cache.keys():
                self.SA_cache[int(label.cpu())] += [SA_batch[i].detach().cpu().numpy()]
            else:
                self.SA_cache[int(label.cpu())] = [SA_batch[i].detach().cpu().numpy()]

    def to_numpy(self):
        for k in self.SA_cache.keys():
            self.SA_cache[k] = np.stack(self.SA_cache[k], 0)

    def set_kde(self):
        raise NotImplementedError
    
    def compute_covinv(self):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError

    def update(self, cove_set, delta=None):
        self.coverage_set = cove_set
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(self.coverage_set)

    def coverage(self, cove_set):
        return len(cove_set)

    def gain(self, cove_set_new):
        new_rate = self.coverage(cove_set_new)
        return new_rate - self.current

    def save(self, path):
        print('Saving recorded %s in %s...' % (self.name, path))
        state = {
            'coverage_set': list(self.coverage_set),
            'mask_index_dict': self.mask_index_dict,
            'mean_dict': self.mean_dict,
            'var_dict': self.var_dict,
            'SA_cache': self.SA_cache
        }
        torch.save(state, path)

    def load(self, path):
        print('Loading saved %s in %s...' % (self.name, path))
        state = torch.load(path)
        self.coverage_set = set(state['coverage_set'])
        self.mask_index_dict = state['mask_index_dict']
        self.mean_dict = state['mean_dict']
        self.var_dict = state['var_dict']
        self.SA_cache = state['SA_cache']
        loaded_cov = self.coverage(self.coverage_set)
        print('Loaded coverage: %f' % loaded_cov)


class NLC(Coverage):
    def init_variable(self, hyper=None):
        assert hyper is None, 'NLC has no hyper-parameter'
        self.estimator_dict = {}
        self.current = 1
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.estimator_dict[layer_name] = tool.Estimator(feature_num=layer_size[0])
    
    def calculate(self, data):
        stat_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            info_dict = self.estimator_dict[layer_name].calculate(layer_output.to(self.device))
            stat_dict[layer_name] = (info_dict['Ave'], info_dict['CoVariance'], info_dict['Amount'])
        return stat_dict

    def update(self, stat_dict, gain=None):
        if gain is None:    
            for i, layer_name in enumerate(stat_dict.keys()):
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave
                self.estimator_dict[layer_name].CoVariance = new_CoVariance
                self.estimator_dict[layer_name].Amount = new_Amount
            self.current = self.coverage(self.estimator_dict)
        else:
            (delta, layer_to_update) = gain
            for layer_name in layer_to_update:
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave
                self.estimator_dict[layer_name].CoVariance = new_CoVariance
                self.estimator_dict[layer_name].Amount = new_Amount
            self.current += delta

    def coverage(self, stat_dict):
        val = 0
        for i, layer_name in enumerate(stat_dict.keys()):
            # Ave = stat_dict[layer_name].Ave
            CoVariance = stat_dict[layer_name].CoVariance
            # Amount = stat_dict[layer_name].Amount
            val += self.norm(CoVariance)
        return val

    def gain(self, stat_new):
        total = 0
        layer_to_update = []
        for i, layer_name in enumerate(stat_new.keys()):
            (new_Ave, new_CoVar, new_Amt) = stat_new[layer_name]
            value = self.norm(new_CoVar) - self.norm(self.estimator_dict[layer_name].CoVariance)
            if value > 0:
                layer_to_update.append(layer_name)
                total += value
        if total > 0:
            return (total, layer_to_update)
        else:
            return None

    def norm(self, vec, mode='L1', reduction='mean'):
        m = vec.size(0)
        assert mode in ['L1', 'L2']
        assert reduction in ['mean', 'sum']
        if mode == 'L1':
            total = vec.abs().sum()
        elif mode == 'L2':
            total = vec.pow(2).sum().sqrt()
        if reduction == 'mean':
            return total / m
        elif reduction == 'sum':
            return total

    def save(self, path):
        print('Saving recorded NLC in %s...' % path)
        stat_dict = {}
        for layer_name in self.estimator_dict.keys():
            stat_dict[layer_name] = {
                'Ave': self.estimator_dict[layer_name].Ave,
                'CoVariance': self.estimator_dict[layer_name].CoVariance,
                'Amount': self.estimator_dict[layer_name].Amount
            }
        torch.save({'stat': stat_dict}, path)

    def load(self, path):
        print('Loading saved NLC from %s...' % path)
        ckpt = torch.load(path)
        stat_dict = ckpt['stat']
        for layer_name in stat_dict.keys():
            self.estimator_dict[layer_name].Ave = stat_dict[layer_name]['Ave']
            self.estimator_dict[layer_name].CoVariance = stat_dict[layer_name]['CoVariance']
            self.estimator_dict[layer_name].Amount = stat_dict[layer_name]['Amount']


class LSC(SurpriseCoverage):
    def get_name(self):
        return 'LSC'

    def set_kde(self):
        for k in self.SA_cache.keys():
            if self.num_class <= 1:
                self.kde_cache[k] = gaussian_kde(self.SA_cache[k].T)
            else:
                self.kde_cache[k] = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(self.SA_cache[k])
            # The original LSC uses the `gaussian_kde` function, however, we note that this function
            # frequently crashes due to numerical issues, especially for large `num_class`.

    def calculate(self, data_batch, label_batch):
        cove_set = set()
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1).detach().cpu().numpy() # [batch_size, num_neuron]
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]
            # if (np.isnan(SA).any()) or (not np.isinf(SA).any()):
            #     continue
            if self.num_class <= 1:
                lsa = np.asscalar(-self.kde_cache[int(label.cpu())].logpdf(np.expand_dims(SA, 1)))
            else:
                lsa = np.asscalar(-self.kde_cache[int(label.cpu())].score_samples(np.expand_dims(SA, 0)))
            if (not np.isnan(lsa)) and (not np.isinf(lsa)):
                cove_set.add(int(lsa / self.threshold))
        cove_set = self.coverage_set.union(cove_set)
        return cove_set


class DSC(SurpriseCoverage):
    def get_name(self):
        return 'DSC'

    def calculate(self, data_batch, label_batch):
        cove_set = set()
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1).detach().cpu().numpy() # [batch_size, num_neuron]
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]
            
            # # using numpy 
            # dist_a_list = np.linalg.norm(SA - self.SA_cache[int(label.cpu())], axis=1)
            # idx_a = np.argmin(dist_a_list, 0)

            dist_a_list = torch.linalg.norm(
                torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[int(label.cpu())]).to(self.device), dim=1)
            idx_a = torch.argmin(dist_a_list, 0).item()

            (SA_a, dist_a) = (self.SA_cache[int(label.cpu())][idx_a], dist_a_list[idx_a])
            dist_a = dist_a.cpu().numpy()

            dist_b_list = []
            for j in range(self.num_class):
                if ( j != int(label.cpu()) ) and ( j in self.SA_cache.keys() ):
                    # # using numpy
                    # dist_b_list += np.linalg.norm(SA - self.SA_cache[j], axis=1).tolist()
                    dist_b_list += torch.linalg.norm(
                        torch.from_numpy(SA).to(self.device) - torch.from_numpy(self.SA_cache[j]).to(self.device),
                        dim=1).cpu().numpy().tolist()
            
            dist_b = np.min(dist_b_list)
            dsa = dist_a / dist_b if dist_b > 0 else 1e-6
            if (not np.isnan(dsa)) and (not np.isinf(dsa)):
                cove_set.add(int(dsa / self.threshold))
        cove_set = self.coverage_set.union(cove_set)
        return cove_set

class MDSC(SurpriseCoverage):
    def get_name(self):
        return 'MDSC'

    def set_mask(self):
        feature_num = 0
        for layer_name in self.mean_dict.keys():
            self.mask_index_dict[layer_name] = (self.var_dict[layer_name] >= self.min_var).nonzero()
            feature_num += self.mask_index_dict[layer_name].size(0)
        print('feature_num: ', feature_num)
        self.estimator = tool.Estimator(feature_num=feature_num, num_class=self.num_class)

    def build_SA(self, data_batch, label_batch):
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1) # [batch_size, num_neuron]
        # print('SA_batch: ', SA_batch.size())
        SA_batch = SA_batch[~torch.any(SA_batch.isnan(), dim=1)]
        SA_batch = SA_batch[~torch.any(SA_batch.isinf(), dim=1)]
        stat_dict = self.estimator.calculate(SA_batch, label_batch)
        self.estimator.update(stat_dict)

    def calculate(self, data_batch, label_batch):
        cove_set = set()
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1) # [batch_size, num_neuron]
        mu = self.estimator.Ave[label_batch]
        covar_inv = self.estimator.CoVarianceInv[label_batch]
        mdsa = (
            torch.bmm(torch.bmm((SA_batch - mu).unsqueeze(1), covar_inv),
            (SA_batch - mu).unsqueeze(2))
        ).sqrt()
        # [bs, 1, n] x [bs, n, n] x [bs, n, 1]
        # [bs, 1]
        mdsa = mdsa.view(batch_size, -1)
        mdsa = mdsa[~torch.any(mdsa.isnan(), dim=1)]
        mdsa = mdsa[~torch.any(mdsa.isinf(), dim=1)]
        mdsa = mdsa.view(-1)
        if len(mdsa) > 0:
            mdsa_list = (mdsa / self.threshold).cpu().numpy().tolist()
            mdsa_list = [int(_mdsa) for _mdsa in mdsa_list]
            cove_set = set(mdsa_list)
            cove_set = self.coverage_set.union(cove_set)
        return cove_set
    
    def compute_covinv(self):
        self.estimator.invert()

    def save(self, path):
        print('Saving recorded %s in %s...' % (self.name, path))
        state = {
            'coverage_set': list(self.coverage_set),
            'mask_index_dict': self.mask_index_dict,
            'mean_dict': self.mean_dict,
            'var_dict': self.var_dict,
            'mu': self.estimator.Ave,
            'covar': self.estimator.CoVariance,
            'amount': self.estimator.Amount
        }
        torch.save(state, path)

    def load(self, path):
        print('Loading saved %s in %s...' % (self.name, path))
        state = torch.load(path)
        self.coverage_set = set(state['coverage_set'])
        self.mask_index_dict = state['mask_index_dict']
        self.mean_dict = state['mean_dict']
        self.var_dict = state['var_dict']
        self.estimator.Ave = state['mu']
        self.estimator.CoVariance = state['covar']
        self.estimator.Amount = state['amount']
        loaded_cov = self.coverage(self.coverage_set)


class NC(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.threshold = hyper
        self.coverage_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.coverage_dict[layer_name] = torch.zeros(layer_size[0]).type(torch.BoolTensor).to(self.device)
        self.current = 0

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            scaled_output = tool.scale(layer_output)
            mask_index = scaled_output > self.threshold
            is_covered = mask_index.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict
    
    def coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()

    def save(self, path):
        print('Saving recorded NC in %s...' % path)
        torch.save(self.coverage_dict, path)

    def load(self, path):
        print('Loading saved NC from %s...' % path)
        self.coverage_dict = torch.load(path)


class KMNC(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.k = int(hyper)
        self.name = 'KMNC'
        self.range_dict = {}
        coverage_multisec_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            coverage_multisec_dict[layer_name] = torch.zeros((num_neuron, self.k + 1)).type(torch.BoolTensor).to(self.device)
            self.range_dict[layer_name] = [torch.ones(num_neuron).to(self.device) * 10000, torch.ones(num_neuron).to(self.device) * -10000]
        self.coverage_dict = {
            'multisec': coverage_multisec_dict
        }
        self.current = 0

    def build(self, data_loader):
        print('Building range...')
        for data, *_ in tqdm(data_loader):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.set_range(data)

    def set_range(self, data):
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * cur_max + ~is_greater * self.range_dict[layer_name][1]

    def calculate(self, data):
        multisec_cove_dict = {}
        lower_cove_dict = {}
        upper_cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            num_neuron = layer_output.size(1)
            multisec_index = (u_bound > l_bound) & (layer_output >= l_bound) & (layer_output <= u_bound)
            multisec_covered = torch.zeros(num_neuron, self.k + 1).type(torch.BoolTensor).to(self.device)
            div_index = u_bound > l_bound
            div = (~div_index) * 1e-6 + div_index * (u_bound - l_bound)
            multisec_output = torch.ceil((layer_output - l_bound) / div * self.k).type(torch.LongTensor).to(self.device) * multisec_index
            # (1, k), index 0 indicates out-of-range output

            index = tuple([torch.LongTensor(list(range(num_neuron))), multisec_output])
            multisec_covered[index] = True
            multisec_cove_dict[layer_name] = multisec_covered | self.coverage_dict['multisec'][layer_name]
        
        return {
            'multisec': multisec_cove_dict
        }

    def coverage(self, cove_dict):
        multisec_cove_dict = cove_dict['multisec']
        (multisec_cove, multisec_total) = (0, 0)
        for layer_name in multisec_cove_dict.keys():
            multisec_covered = multisec_cove_dict[layer_name]
            num_neuron = multisec_covered.size(0)
            multisec_cove += torch.sum(multisec_covered[:, 1:])
            multisec_total += (num_neuron * self.k)
        multisec_rate = multisec_cove / multisec_total
        return multisec_rate.item()

    def save(self, path):
        print('Saving recorded %s in \%s' % (self.name, path))
        state = {
            'range': self.range_dict,
            'coverage': self.coverage_dict
        }
        torch.save(state, path)

    def load(self, path):
        print('Loading saved %s from %s' % (self.name, path))
        state = torch.load(path)
        self.range_dict = state['range']
        self.coverage_dict = state['coverage']


class SNAC(KMNC):
    def init_variable(self, hyper=None):
        assert hyper is None
        self.name = 'SNAC'
        self.range_dict = {}
        coverage_upper_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            coverage_upper_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
            self.range_dict[layer_name] = [torch.ones(num_neuron).to(self.device) * 10000, torch.ones(num_neuron).to(self.device) * -10000]
        self.coverage_dict = {
            'upper': coverage_upper_dict
        }
        self.current = 0

    def calculate(self, data):
        upper_cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            num_neuron = layer_output.size(1)
            upper_covered = (layer_output > u_bound).sum(0) > 0
            upper_cove_dict[layer_name] = upper_covered | self.coverage_dict['upper'][layer_name]

        return {
            'upper': upper_cove_dict
        }

    def coverage(self, cove_dict):
        upper_cove_dict = cove_dict['upper']
        (upper_cove, upper_total) = (0, 0)
        for layer_name in upper_cove_dict.keys():
            upper_covered = upper_cove_dict[layer_name]
            upper_cove += upper_covered.sum()
            upper_total += len(upper_covered)
        upper_rate = upper_cove / upper_total
        return upper_rate.item()


class NBC(KMNC):
    def init_variable(self, hyper=None):
        assert hyper is None
        self.name = 'NBC'
        self.range_dict = {}
        coverage_lower_dict = {}
        coverage_upper_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            coverage_lower_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
            coverage_upper_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
            self.range_dict[layer_name] = [torch.ones(num_neuron).to(self.device) * 10000, torch.ones(num_neuron).to(self.device) * -10000]
        self.coverage_dict = {
            'lower': coverage_lower_dict,
            'upper': coverage_upper_dict
        }
        self.current = 0

    def calculate(self, data):
        lower_cove_dict = {}
        upper_cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            
            lower_covered = (layer_output < l_bound).sum(0) > 0
            upper_covered = (layer_output > u_bound).sum(0) > 0

            lower_cove_dict[layer_name] = lower_covered | self.coverage_dict['lower'][layer_name]
            upper_cove_dict[layer_name] = upper_covered | self.coverage_dict['upper'][layer_name]
        
        return {
            'lower': lower_cove_dict,
            'upper': upper_cove_dict
        }

    def coverage(self, cove_dict):
        lower_cove_dict = cove_dict['lower']
        upper_cove_dict = cove_dict['upper']

        (lower_cove, lower_total) = (0, 0)
        (upper_cove, upper_total) = (0, 0)
        for layer_name in lower_cove_dict.keys():
            lower_covered = lower_cove_dict[layer_name]
            upper_covered = upper_cove_dict[layer_name]
            
            lower_cove += lower_covered.sum()
            upper_cove += upper_covered.sum()
            
            lower_total += len(lower_covered)
            upper_total += len(upper_covered)
        lower_rate = lower_cove / lower_total
        upper_rate = upper_cove / upper_total
        return (lower_rate + upper_rate).item() / 2


class TKNC(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.k = int(hyper)
        self.coverage_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            self.coverage_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
        self.current = 0

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            batch_size = layer_output.size(0)
            num_neuron = layer_output.size(1)
            # layer_output: (batch_size, num_neuron)
            _, idx = layer_output.topk(min(self.k, num_neuron), dim=1, largest=True, sorted=False)
            # idx: (batch_size, k)
            covered = torch.zeros(layer_output.size()).to(self.device)
            index = tuple([torch.LongTensor(list(range(batch_size))), idx.transpose(0, 1)])
            covered[index] = 1
            is_covered = covered.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict

    def coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()

    def save(self, path):
        print('Saving recorded TKNC in %s' % path)
        torch.save(self.coverage_dict, path)

    def load(self, path):
        print('Loading saved TKNC from %s' % path)
        self.coverage_dict = torch.load(path)


class TKNP(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.k = hyper
        layer_pattern = {}
        network_pattern = set()
        self.current = 0
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.layer_pattern[layer_name] = set()
        self.coverage_dict = {
            'layer_pattern': layer_pattern,
            'network_pattern': network_pattern
        }

    def calculate(self, data):
        layer_pat = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        topk_idx_list = []
        for (layer_name, layer_output) in layer_output_dict.items():
            num_neuron = layer_output.size(1)
            _, idx = layer_output.topk(min(self.k, num_neuron), dim=1, largest=True, sorted=True)
            # idx: (batch_size, k)
            pat = set([str(s) for s in list(idx[:, ])])
            topk_idx_list.append(idx)
            layer_pat[layer_name] = set.union(pat, self.layer_pattern[layer_name])
        network_topk_idx = torch.cat(topk_idx_list, 1)
        network_pat = set([str(s) for s in list(network_topk_idx[:, ])])
        network_pat = set.union(network_pat, self.network_pattern)
        return {
            'layer_pattern': layer_pat,
            'network_pattern': network_pat
        }

    def coverage(self, cove_dict, mode='network'):
        assert mode in ['network', 'layer']
        if mode == 'network':
            return len(cove_dict['network_pattern'])
        cnt = 0
        if mode == 'layer':
            for layer_name in cove_dict['layer_pattern'].keys():
                cnt += len(cove_dict['layer_pattern'][layer_name])
        return cnt

    def save(self, path):
        print('Saving recorded TKNP in %s' % path)
        torch.save(self.coverage_dict, path)

    def load(self, path):
        print('Loading saved TKNP from %s' % path)
        self.coverage_dict = torch.load(path)


class CC(Coverage):
    '''
    Cluster-based Coverage, i.e., the coverage proposed by TensorFuzz
    '''
    def init_variable(self, hyper):
        assert hyper is not None
        self.threshold = hyper
        self.distant_dict = {}
        self.flann_dict = {}

        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.flann_dict[layer_name] = FLANN()
            self.distant_dict[layer_name] = []

    def update(self, dist_dict, delta=None):
        for layer_name in self.distant_dict.keys():
            self.distant_dict[layer_name] += dist_dict[layer_name]
            self.flann_dict[layer_name].build_index(np.array(self.distant_dict[layer_name]))
        if delta:
            self.current += delta
        else:
            self.current += self.coverage(dist_dict)

    def calculate(self, data):
        layer_output_dict = tool.get_layer_output(self.model, data)
        dist_dict = {}
        for (layer_name, layer_output) in layer_output_dict.items():
            dist_dict[layer_name] = []
            for single_output in layer_output:
                single_output = single_output.cpu().numpy()
                if len(self.distant_dict[layer_name]) > 0:
                    _, approx_distances = self.flann_dict[layer_name].nn_index(np.expand_dims(single_output, 0), num_neighbors=1)
                    exact_distances = [
                        np.sum(np.square(single_output - distant_vec))
                        for distant_vec in self.distant_dict[layer_name]
                    ]
                    buffer_distances = [
                        np.sum(np.square(single_output - buffer_vec))
                        for buffer_vec in dist_dict[layer_name]
                    ]
                    nearest_distance = min(exact_distances + approx_distances.tolist() + buffer_distances)
                    if nearest_distance > self.threshold:
                        dist_dict[layer_name].append(single_output)
                else:
                    self.flann_dict[layer_name].build_index(single_output)
                    self.distant_dict[layer_name].append(single_output)
        return dist_dict

    def coverage(self, dist_dict):
        total = 0
        for layer_name in dist_dict.keys():
            total += len(dist_dict[layer_name])
        return total

    def gain(self, dist_dict):
        increased = self.coverage(dist_dict)
        return increased

    def save(self, path):
        print('Saving recorded CC in %s' % path)
        torch.save(self.distant_dict, path)

    def load(self, path):
        print('Loading saved CC from %s' % path)
        self.distant_dict = torch.load(path)


if __name__ == '__main__':
    pass