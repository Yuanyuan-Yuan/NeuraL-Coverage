import torch
import torch.nn as nn
import torch.nn.functional as F

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