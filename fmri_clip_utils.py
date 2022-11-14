import math
import time
import h5py
import itertools
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import ipdb
import wandb
from tqdm import tqdm

import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torch.autograd import Variable

from nsd_utils import *

def viz_clip(idx, dataset, model, vec_key='clip_img', viz_len=None):
    ''' Plot pred and gt of idx-th sample on a same figure.'''
    x = dataset[idx]

    model.eval()
    with torch.no_grad():
        y = model(x, return_loss=False)

    viz_len = len(y.squeeze()) if viz_len is None else viz_len
    plt.figure()
    plt.plot(x[vec_key].squeeze().cpu().numpy()[:viz_len], label='gt')
    plt.plot(y.squeeze().cpu().numpy()[:viz_len], label='pred')        
    plt.legend()
    plt.show()

def viz_clip2(idxs, dataset, model, vec_key='clip_img', gt=False, viz_len=None):
    ''' Plot multiple predictions/ gt on a same figure.
    - gt: bool. If True, the figure will plot gt.
    '''
    model.eval()
    plt.figure()
    for idx in idxs:
        x = dataset[idx]

        with torch.no_grad():
            y = model(x, return_loss=False)

        viz_len = len(y.squeeze()) if viz_len is None else viz_len
        if gt:
            plt.plot(x[vec_key].squeeze().cpu().numpy()[:viz_len], label=str(idx))
        else:
            plt.plot(y.squeeze().cpu().numpy()[:viz_len], label=str(idx))
        plt.legend()
    plt.show()

def print_eval(x, sim):
    b = len(sim)
    print('======================')
    c1 = (x['nsdId'][np.argmax(sim, 0)] == x['nsdId'][np.arange(b)]).sum()
    print(f'which clip is the closest to the pred? Correct match is {c1}',
          f'out of {b}')
    c2 = (x['nsdId'][np.argmax(sim, 1)] == x['nsdId'][np.arange(b)]).sum()
    print(f'which pred is the closest to the clip? Correct match is {c2}',
          f'out of {b}')
    return c1, c2

def show_viz(x, sim, viz_idx=0):
    sorted_idx = np.argsort(sim[viz_idx])[::-1]
    viz_list = list(sorted_idx[: np.where(sorted_idx == viz_idx)[0][0] + 1])
    print(f'there are {len(viz_list) - 1} mapped-fmri being closer to the',
          'clip vec than the gt mapped-fmri as the following.')

    # check coco id, and load image to check
    with open(STIM_INFO, 'rb') as f:
        stim_info = pickle.load(f, encoding='latin1')
    
    cocoIds = []
    with h5py.File(STIM_FILE, 'r') as f:
        for i in viz_list:
            nsdId = x['nsdId'][i].item()
            cocoId = stim_info['cocoId'][nsdId]
            print(f"\nnsd id: {nsdId}; coco id: {cocoId}")
            cocoIds.append(cocoId)

            img_sample = f['imgBrick'][nsdId]
            plt.imshow(img_sample)
            plt.xticks([])
            plt.yticks([])
            plt.show()

    print(f'In total {len(set(cocoIds)) - 1} unique images before gt.')

def sim_eval(model, x, vec_key='clip_img', cos_weight=1.0, viz_idx=0):
    '''
    - model: the model matching fmri to clip image vector
    - x: one batch of the dataloader, should has key 'fmri', vec_key, 'nsdId'
    - cos_weight: the evaluation will based on a combination of cos similarity
                  ans negative L2 distance, this weight specifies the ratio.
    - viz_idx: for visualizing which pred is closest to the viz_idx clip vector
    '''
    model.eval()
    clip_img = x[vec_key]
    with torch.no_grad():
        pred = model(x)

    b = len(clip_img) # batch_size
    sim = torch.zeros((b, b))
    for i in tqdm(range(b)):
        for j in range(b):
            sim[i, j] = (F.cosine_similarity(clip_img[i], pred[j], -1) * 
                         cos_weight - (1 - cos_weight) * 
                         F.mse_loss(clip_img[i], pred[j]))

    sim = sim.cpu().numpy()
    plt.imshow(sim)
    plt.show()

    print_eval(x, sim)
    show_viz(x, sim, viz_idx=viz_idx)

    return sim

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def exclude_bias_and_norm(p):
    return p.ndim == 1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class cos_sim_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss()

    def forward(self, y_pred, y):
        target = torch.ones(len(y)).to(y.device)
        return self.loss_fn(y_pred, y, target)


class mse_cos_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.cos_loss = cos_sim_loss()

    def forward(self, y_pred, y, cos_weight):
        return ((1 - cos_weight) * self.mse_loss(y_pred, y) +
                cos_weight * self.cos_loss(y_pred, y))


class contrastive_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y, temperature=0.5, lam=0.5, cos_weight=None):
        # _dot = torch.bmm(y.view(len(y), 1, -1),
        #     y_pred.view(len(y_pred), -1, 1)).squeeze()
        # _norm = torch.norm(y, dim=1) * torch.norm(y_pred, dim=1)
        # cos_sim = _dot/ _norm # shape (32,)
        # loss = -F.log_softmax(cos_sim / temperature, dim=0)
        # return torch.mean(loss)

        sim = torch.cosine_similarity(y_pred.unsqueeze(1), y.unsqueeze(0), dim=-1)
        # sim: shape (32, 32), diagonal is equivalant to above cos_sim
        if temperature > 0.:
            sim = sim / temperature
            # the above loss = - F.log_softmax(torch.diagonal(sim), dim=0)
            # whereas the below loss = - torch.diagonal(F.log_softmax(sim, dim=0))
            sim1 = torch.diagonal(F.log_softmax(sim, dim=1))
            sim2 = torch.diagonal(F.log_softmax(sim, dim=0))
            return (-(lam * sim1 + (1. - lam) * sim2)).mean()
        else:
            return (-torch.diagonal(sim)).mean()


class mse_cos_contrastive_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_cos_loss = mse_cos_loss()
        self.contrastive_loss = contrastive_loss()
    def forward(self, y_pred, y, temperature=0.5, cos_weight=0.5, contra_p=0.5):
        return ((1 - contra_p) * self.mse_cos_loss(y_pred, y, cos_weight) +
            contra_p * self.contrastive_loss(y_pred, y, temperature))


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.0,
                        torch.where(update_norm > 0,
                            (g["eta"] * param_norm / update_norm), one), one,)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])
