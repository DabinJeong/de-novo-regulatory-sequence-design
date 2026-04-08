# Adapted from https://github.com/HannesStark/dirichlet-flow-matching/blob/main/utils/flow_utils.py
import copy
import math
import pickle

import scipy
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm

class DirichletConditionalFlow:
    def __init__(self, K=20, alpha_min=1, alpha_max=100, alpha_spacing=0.01):
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing
        self.K = K

    def c_factor(self, bs, alpha):
        out1 = scipy.special.beta(alpha, self.K - 1)
        out2 = np.where(bs < 1, out1 / ((1 - bs) ** (self.K - 1)), 0)
        out = np.where((bs ** (alpha - 1)) > 0, out2 / (bs ** (alpha - 1)), 0)
        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, I_func)
        final = interp * out
        return final

def expand_simplex(xt, alphas, prior_pseudocount):
    prior_weights = (prior_pseudocount / (alphas + prior_pseudocount - 1))[:, None, None]
    return torch.cat([xt * (1 - prior_weights), xt * prior_weights], -1), prior_weights

def sample_cond_prob_path(args, seq, alphabet_size):
    B, L = seq.shape
    seq_one_hot = torch.nn.functional.one_hot(seq, num_classes=alphabet_size)
    if args.mode == 'dirichlet':
        alphas = torch.from_numpy(1 + scipy.stats.expon().rvs(size=B) * args.alpha_scale).to(seq.device).float()
        if args.fix_alpha:
            alphas = torch.ones(B, device=seq.device) * args.fix_alpha
        alphas_ = torch.ones(B, L, alphabet_size, device=seq.device)
        alphas_ = alphas_ + seq_one_hot * (alphas[:,None,None] - 1)
        xt = torch.distributions.Dirichlet(alphas_).sample()
    elif args.mode == 'distill':
        alphas = torch.zeros(B, device=seq.device)
        xt = torch.distributions.Dirichlet(torch.ones(B, L, alphabet_size, device=seq.device)).sample()
    elif args.mode == 'riemannian':
        t = torch.rand(B, device=seq.device)
        dirichlet = torch.distributions.Dirichlet(torch.ones(alphabet_size, device=seq.device))
        x0 = dirichlet.sample((B,L))
        x1 = seq_one_hot
        xt = t[:,None,None] * x1 + (1 - t[:,None,None]) * x0
        alphas = t
    elif args.mode == 'ardm' or args.mode == 'lrar':
        mask_prob = torch.rand(1, device=seq.device)
        mask = torch.rand(seq.shape, device=seq.device) < mask_prob
        if args.mode == 'lrar': mask = ~(torch.arange(L, device=seq.device) < (1-mask_prob) * L)
        xt = torch.where(mask, alphabet_size, seq) # mask token index
        xt = torch.nn.functional.one_hot(xt, num_classes=alphabet_size + 1).float() # plus one to include index for mask token
        alphas = mask_prob.expand(B)
    return xt, alphas

def simplex_proj(seq):
    """Algorithm from https://arxiv.org/abs/1309.1541 Weiran Wang, Miguel Á. Carreira-Perpiñán"""
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(seq.shape)

def get_wasserstein_dist(embeds1, embeds2):
    if np.isnan(embeds2).any() or np.isnan(embeds1).any() or len(embeds1) == 0 or len(embeds2) == 0:
        return float('nan')
    mu1, sigma1 = embeds1.mean(axis=0), np.cov(embeds1, rowvar=False)
    mu2, sigma2 = embeds2.mean(axis=0), np.cov(embeds2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    dist = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return dist

def update_ema(current_dict, prev_ema, gamma = 0.9):
    ema = copy.deepcopy(prev_ema)
    current_dict = copy.deepcopy(current_dict)
    for key, current_value in current_dict.items():
        ema_key  = 'ema_' + key
        if not np.isnan(current_value):
            if ema_key in prev_ema:
                ema[ema_key] = (1 - gamma) * current_value + gamma * prev_ema[ema_key]
            else:
                ema[ema_key] = current_value
    return ema

def load_flybrain_designed_seqs(path):
    order = {'A': 0, 'C':1, 'G':2, 'T':3}
    f = open(path, "rb")
    data = pickle.load(f)
    arrays = []
    for seq in data['seq']:
        arrays.append([order[char] for char in seq])
    return torch.tensor(arrays, dtype=torch.long)