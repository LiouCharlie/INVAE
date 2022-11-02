from scipy.stats import f
import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable

def anova_loss(data,class_0,class_1):
    eps = 10**-45
    num_class_0 = torch.unique(class_0)
    num_class_1 = torch.unique(class_1)
    tot_loss = 0
    count_break = -1
    
    for i in num_class_0:
        marg_mu = data[(class_0 == i).squeeze()].mean(0)
        SSR = 0
        SSE = 0
        SSR_org = 0
        SSE_org = 0
        

        if len(torch.unique(class_1[class_0==i])) >1:
            num_sub_class = len(torch.unique(class_1[class_0==i]))
            for j in num_class_1:
                mask = ((class_0==i) & (class_1==j)).squeeze()
                if mask.sum() != 0:
                    SSR_temp = (mask.sum())*(data[mask].mean(0) - marg_mu)**2
                    SSE_temp = (data[mask] - data[mask].mean(0))**2
                    SSR += SSR_temp
                    SSE += SSE_temp.sum(0)
                    
            d1 = ((class_0 == i).squeeze().sum() - num_sub_class).item()
            d2 = (num_sub_class-1)
            temp_loss_raw = (SSR * ((class_0 == i).squeeze().sum() - num_sub_class)) / ((SSE*(num_sub_class-1)) + eps)
            
            temp_loss = torch.abs(temp_loss_raw)
            tot_loss += temp_loss
    return tot_loss

class Normal(nn.Module):
    """Function proposed in original beta-TCVAE paper(Ricky T. Q. Chen et al, NIPS, 2018)."""

    def __init__(self, mu=0, sigma=1):
        super(Normal, self).__init__()
        self.normalization = Variable(torch.Tensor([np.log(2 * np.pi)]))

        self.mu = Variable(torch.Tensor([mu]))
        self.logsigma = Variable(torch.Tensor([math.log(sigma)]))

    def _check_inputs(self, size, mu_logsigma):
        if size is None and mu_logsigma is None:
            raise ValueError(
                'Either one of size or params should be provided.')
        elif size is not None and mu_logsigma is not None:
            mu = mu_logsigma.select(-1, 0).expand(size)
            logsigma = mu_logsigma.select(-1, 1).expand(size)
            return mu, logsigma
        elif size is not None:
            mu = self.mu.expand(size)
            logsigma = self.logsigma.expand(size)
            return mu, logsigma
        elif mu_logsigma is not None:
            mu = mu_logsigma.select(-1, 0)
            logsigma = mu_logsigma.select(-1, 1)
            return mu, logsigma
        else:
            raise ValueError(
                'Given invalid inputs: size={}, mu_logsigma={})'.format(
                    size, mu_logsigma))

    def sample(self, size=None, params=None):
        mu, logsigma = self._check_inputs(size, params)
        std_z = Variable(torch.randn(mu.size()).type_as(mu.data))
        sample = std_z * torch.exp(logsigma) + mu
        return sample

    def log_density(self, sample, params=None):
        if params is not None:
            mu, logsigma = self._check_inputs(None, params)
        else:
            mu, logsigma = self._check_inputs(sample.size(), None)
            mu = mu.type_as(sample)
            logsigma = logsigma.type_as(sample)

        c = self.normalization.type_as(sample.data)
        inv_sigma = torch.exp(-logsigma)
        tmp = (sample - mu) * inv_sigma

        return -0.5 * (tmp * tmp + 2 * logsigma + c)

    
def logsumexp(value, dim=None, keepdim=False):
    """Function proposed in original beta-TCVAE paper(Ricky T. Q. Chen et al, NIPS, 2018)."""
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)