import os
rt_dir = '/home/sikun/bold5k/'
os.chdir(rt_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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
from modules import *


def adjust_learning_rate(epochs, base_lr, optimizer, loader, step):
    max_steps = epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = base_lr * loader.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def updateG(model, x, vec_key, is_train=True, optimizerG=None, netD=None,
            lossD=None, optimizerD=None, prev_errD=0., threshold_D=0.5,
            grad_clip_norm=None, cos_weight_override=None):
    ''' Update fusion model (the generator, if using GAN).'''
    errD = None
    errG = None

    if netD:
        real_label = 1.
        label = torch.full((len(x[vec_key]),), real_label, dtype=torch.float,
                           device=device)
        lossD_wts = 1. #10., put more weight on GAN loss, to combat errD becoming 0 fast

    if is_train:
        y, loss = model(x, return_loss=True)
        if netD: # GAN
            vec = x[vec_key] # for passing to discriminator
            # update discriminator
            # only update D when loss is larger than a certain value
            # errD = updateD(netD, _y, vec, lossD, optimizerD) if (
            #     prev_errD > threshold_D) else torch.rand_like(prev_errD)
            errD = updateD(netD, y, vec, lossD, optimizerD)
        model.zero_grad()
        loss.backward()

        if netD:
            # add discriminator loss if using GAN
            output = netD(y).view(-1)
            errG = lossD(output, label) * lossD_wts
            errG.backward()

        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizerG.step()        

    else:
        with torch.no_grad():
            if cos_weight_override is not None:
                y, loss = model(x, return_loss=True,
                                cos_weight_override=cos_weight_override)
            else:
                y, loss = model(x, return_loss=True)
            if netD:
                output = netD(y).view(-1)
                errG = lossD(output, label) * lossD_wts
                vec = x[vec_key]
                errD = updateD(netD, y, vec, lossD, is_train=is_train)

    return loss, errD, errG

def updateD(netD, y, vec, lossD, optimizerD=None, is_train=True):
    ''' if using GAN, this function will be used to update discriminator net.'''
    ''' Discriminator for image code.'''
    real_label = 1.
    fake_label = 0.
    label = torch.full((len(y),), real_label, dtype=torch.float, device=device)

    if is_train:
        # real batch: use vec (gt)
        netD.zero_grad()
        # Forward pass real batch through D
        output = netD(vec).view(-1)
        # Calculate loss on all-real batch
        errD_real = lossD(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        # D_x = output.mean().item()

        # fake batch: use y
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(y.detach()).view(-1) # <-- output = netD(y).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = lossD(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        # D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake

        # Update D
        optimizerD.step()

    else:
        with torch.no_grad():
            output = netD(vec).view(-1)
            errD = lossD(output, label)
            label.fill_(fake_label)
            output = netD(y).view(-1)
            errD += lossD(output, label)

    return errD

def train_fmri_clip(model, epochs, learning_rate, weight_decay, grad_clip_norm,
                    train_loader, val_loader, vec_key='clip_img',
                    optimizer=None, base_lr=0.0, cos_weight_override=None,
                    lr_decay_epoch=10, netD=None, print_every=50,
                    save_model=None, save_all=False, save_netD=None,
                    min_vloss=None, viz_loss=True, use_wandb=False):
    ''' Train the fmri to vector mapper.
    - model: the mapper model. Loss should be handled by model.
    - epochs: int, training epochs.
    - learning_rate: for optimizer.
    - weight_decay: for optimizer.
    - grad_clip_norm: gradient clip value.
    - train_loader: torch dataloader, for training.
    - val_loader: torch dataloader, for validation.
    - vec_key: x[vec_key] should be the mapped target.
    - optimizer: if None, then will be set to Adam.
    - base_lr: base_lr for LARS.
    - cos_weight_override: for fair eval over sweep runs, can chhose to fix
                           cos_weight during validation.
    - lr_decay_epoch: lr changes per these epochs.
    - netD: if using GAN, pass discriminator here.
    - print_every: print current iteration's loss per these many iters.
    - save_model: pass saving filename with directory. Str.
    - save_all: bool. if True, all models with decreased vloss are saved.
                Model save name will be save_model with the loss attached.
    - save_netD: the model name for saving discriminator if any.
    - min_vloss: current min validation loss, useful when doing warmstart.
    - viz_loss: if True, plot train and val loss after training.
    - use_wandb: whether to use wandb for logging.
    '''
    if use_wandb: wandb.watch(model, log_freq=500)

    min_vloss = float('Inf') if (min_vloss is None) else min_vloss

    isAdam = True if (optimizer is None) else False
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay) if optimizer is None else optimizer
    print(isAdam, optimizer)

    lossD = None
    optimizerD = None
    if netD: # using GAN
        beta1 = 0.5      
        optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate,
                                      betas=(beta1, 0.999))
        lossD = nn.BCEWithLogitsLoss()

    train_loss = []
    val_loss = []
    if netD:
        train_errD = []
        train_errG = []
        val_errD = []
        val_errG = []
    for epoch in range(epochs):
        print('Epoch', epoch)
        # lr decay
        if isAdam and epoch % lr_decay_epoch == 0 and epoch > 0:
            learning_rate *= 0.8
            print(f'learning rate set to {learning_rate}.')
            optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                   weight_decay=weight_decay)
        if use_wandb: wandb.log({'lr': learning_rate})

        ###### Training ######
        model.train()
        ctr = 0
        cur_loss = []
        if netD:
            cur_errD = []
            cur_errG = []
        t_0 = time.time()
        # for step, x in enumerate(train_loader, start=epoch * len(train_loader)):
        for x in train_loader:
            if not isAdam:
                learning_rate = adjust_learning_rate(epochs, base_lr, optimizer,
                                                     train_loader, step)
                if use_wandb: wandb.log({'lr': learning_rate})

            loss, errD, errG = updateG(model, x, vec_key=vec_key, optimizerG=optimizer,
                                       netD=netD, lossD=lossD, optimizerD=optimizerD,
                                       grad_clip_norm=None,)#, prev_errD=cur_errD[-1])

            cur_loss.append(loss)
            if use_wandb: wandb.log({"loss": loss})

            if netD:
                cur_errD.append(errD)
                cur_errG.append(errG)
                if use_wandb: wandb.log({"errD": errD, "errG": errG})

            if ctr % print_every == 0:
                print(f'iter {ctr}: loss {loss.detach().cpu().numpy()}')
                if netD:
                    print(f'\terrD: {errD.detach().cpu().numpy()}',
                          f'errG: {errG.detach().cpu().numpy()}')

            ctr += 1
        train_time = time.time() - t_0
        train_loss.append(sum(cur_loss) / len(cur_loss))

        ###### Validation ######
        model.eval()
        if netD: netD.eval()
        cur_loss = []
        if netD:
            cur_errD = []
            cur_errG = []
        for x in val_loader:
            loss, errD, errG = updateG(model, x, vec_key=vec_key,
                                       is_train=False, netD=netD, lossD=lossD,
                                       cos_weight_override=cos_weight_override)

            cur_loss.append(loss.cpu().numpy())
            if netD:
                cur_errD.append(errD.cpu().numpy())
                cur_errG.append(errG.cpu().numpy())

        val_loss.append(sum(cur_loss) / len(cur_loss))
        if netD:
            val_errD.append(sum(cur_errD) / len(cur_errD))
            val_errG.append(sum(cur_errG) / len(cur_errG))

        ###### Print and Save ######
        print(f'Epoch: {epoch:03d},',
              f'Train Loss: {train_loss[-1].detach().cpu().numpy():.6f},',
              f'Valid Loss: {val_loss[-1]:.6f},',
              f'Training Time: {train_time:.3f}/epoch')
        if netD:
            print(f'Valid errD: {val_errD[-1]:.6f},',
                  f'Valid errG: {val_errG[-1]:.6f}')

        if use_wandb:
            wandb.log({
                'train_loss': train_loss[-1].detach().cpu().numpy(),
                'val_loss': val_loss[-1],
                'epoch time': train_time
                })
            if netD: wandb.log({"val_errD": val_errD[-1],
                                "val_errG": val_errG[-1]})

        if val_loss[-1] < min_vloss:
            min_vloss = val_loss[-1]
            if use_wandb: wandb.log({'min_vloss': min_vloss,})
            if save_model:
                _save_model = (os.path.splitext(save_model)[0] +
                    f'_{min_vloss}' + os.path.splitext(save_model)[1]) if (
                    save_all) else save_model
                torch.save(model.state_dict(), _save_model)
                print(f'Model {_save_model} saved.')
            if save_netD:
                torch.save(netD.state_dict(), save_netD)
                print(f'NetD {save_netD} saved.')

    if viz_loss:
        plt.figure()
        plt.plot([tl.detach().cpu().numpy() for tl in train_loss])
        plt.plot(val_loss)
        plt.show()
        if netD:
            plt.figure()
            plt.plot([tl.detach().cpu().numpy() for tl in train_errD])
            plt.plot([tl.detach().cpu().numpy() for tl in train_errG])
            plt.show()

    if netD:
        return train_loss, val_loss, train_errD, train_errG
    else:
        return train_loss, val_loss


class fmri_clip_2fc(nn.Module):
    def __init__(self, fmri_len=0, start_dim=0, hid_dim=0, vqvae=None,
                 loss_type='contrastive', cos_weight=0.5, vec_key='clip_img'):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(start_dim, hid_dim),
            nn.Linear(hid_dim, 512) # 512 is CLIP dimension
        )
        self.fmri_len = fmri_len
        self.vqvae = vqvae
        if vqvae is not None:
            self.num_tokens = vqvae.num_tokens

        if loss_type == 'cos':
            self.loss_fn = cos_sim_loss()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        if loss_type == 'mse_cos':
            self.loss_fn = mse_cos_loss(cos_weight=cos_weight)
        self.loss_type = loss_type
        self.vec_key = vec_key

    def forward(self, x, return_loss=False, temperature=0.5):
        fmri = x['fmri']
        if self.vqvae is not None:
            fmri = fmri.view(-1, 1, self.fmri_len)
            fmri = self.vqvae.get_codebook_indices(fmri).float()
            # normalize tokens to [-1, 1]
            fmri /= (self.num_tokens / 2)
            fmri -= 1.0


        y_pred = self.linear(fmri)

        if not return_loss:
            return y_pred

        y = x[self.vec_key]
        if self.loss_type == 'contrastive':
            # cosine similarity but contrastive
            _dot = torch.bmm(y.view(len(y), 1, 512),
                            y_pred.view(len(y_pred), 512, 1)).squeeze()
            _norm = torch.norm(y, dim=1) * torch.norm(y_pred, dim=1)                             
            cos_sim = _dot/ _norm
            loss = -F.log_softmax(cos_sim / temperature, dim=0)
            return torch.mean(loss)

        else:
            return self.loss_fn(y_pred, y)


class fmri_clip_res(nn.Module):
    ''' MAIN MAPPER FUNCTION USED IN THE FINAL PAPER'''
    def __init__(
        self,
        signal_len = 256,
        num_tokens = 512,
        num_layers = 1,
        num_resnet_blocks = 4,
        hidden_dim = 64,
        fc_hdim = 128,
        channels = 1,
        last_activation = True,
        loss_type = 'mse_cos',
        cos_weight = 0.5,
        normalization = None,
        vec_key = 'clip_img',
    ):
        super().__init__()
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'

        self.signal_len = signal_len
        self.num_tokens = num_tokens
        self.num_layers = num_layers

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        enc_chans = [channels, *enc_chans]
        enc_chans_io, = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans,))

        enc_layers = []
        for (enc_in, enc_out) in enc_chans_io:
            enc_layers.append(nn.Sequential(nn.Conv1d(enc_in, enc_out, 4,
                                            stride=2, padding=1), nn.ReLU()))

        d = enc_chans[-1]
        # Using no-bottleneck resblock
        for _ in range(num_resnet_blocks):
            enc_layers.append(ResBlock_1d_BN(d))
        enc_layers.append(nn.Conv1d(d, num_tokens, 1))

        # # Using bottleneck resblock
        # if num_resnet_blocks > 0:
        #     enc_layers.append(ResBlock_1d_bottleneck([d, d, d, 4 * d]))
        # for _ in range(num_resnet_blocks - 1):
        #     enc_layers.append(ResBlock_1d_bottleneck([4 * d, d, d, 4 * d]))
        # enc_layers.append(nn.Conv1d(4 * d, num_tokens, 1))

        self.encoder = nn.Sequential(*enc_layers)

        fc = [
              nn.ReLU(),
              nn.Conv1d(signal_len // (2 ** num_layers), fc_hdim, 1),
              nn.ReLU(),
              nn.Conv1d(fc_hdim, 1, 1),
              ]
        if last_activation:
            fc.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc)

        # take care of normalization within class
        self.normalization = normalization
        self._register_external_parameters()

        self.cos_weight = cos_weight
        if loss_type == 'cos':
            self.loss_fn = cos_sim_loss()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        if loss_type == 'mse_cos':
            self.loss_fn = mse_cos_loss()
        if loss_type == 'contrastive':
            self.loss_fn = contrastive_loss()
        if loss_type == 'mse_cos_contrastive':
            self.loss_fn = mse_cos_contrastive_loss()
        if loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        self.loss_type = loss_type
        self.vec_key = vec_key

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    def norm(self, signals):
        if not exists(self.normalization):
            return signals

        means, stds = map(lambda t: torch.as_tensor(t).to(signals), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c ()'), (means, stds))
        signals = signals.clone()
        signals.sub_(means).div_(stds)
        return signals

    def forward(
        self,
        x,
        return_loss = False,
        return_vec = True,
        temperature = 0.5,
        cos_weight_override = None,
        ):
        signal = x['fmri']
        device, num_tokens, signal_len = signal.device, self.num_tokens, self.signal_len
        cos_weight = default(cos_weight_override, self.cos_weight)

        assert signal.shape[-1] == signal_len, f'input must have the correct image size {signal_len}'
        
        signal = signal.view(-1, 1, self.signal_len)
        signal = self.norm(signal)
        signal = self.encoder(signal)
        signal = signal.transpose(1,2)
        y_pred = self.fc(signal)[:, 0, :]

        if not return_loss:
            return y_pred
        
        y = x[self.vec_key]

        if 'contrastive' in self.loss_type:
            if return_vec:
                return y, self.loss_fn(y_pred, y, temperature, cos_weight)
            return self.loss_fn(y_pred, y, temperature, cos_weight)
        elif self.loss_type == 'mse_cos':
            if return_vec:
                return y, self.loss_fn(y_pred, y, cos_weight)
            return self.loss_fn(y_pred, y, cos_weight)
        else: # mse, cos
            if return_vec:
                return y, self.loss_fn(y_pred, y)
            return self.loss_fn(y_pred, y)


class fmri_cat(nn.Module):
    def __init__(
        self,
        signal_len = 256,
        num_class = 80,
        num_layers = 1,
        num_resnet_blocks = 4,
        hidden_dim = 64,
        fc_hdim = 128,
        channels = 1,
        input_key = 'fmri',
    ):
        super().__init__()
        # assert log2(signal_len).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'

        self.signal_len = signal_len
        self.num_class = num_class
        self.num_layers = num_layers

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        enc_chans = [channels, *enc_chans]
        enc_chans_io, = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans,))

        enc_layers = []
        for (enc_in, enc_out) in enc_chans_io:
            enc_layers.append(nn.Sequential(nn.Conv1d(enc_in, enc_out, 4,
                                            stride=2, padding=1), nn.ReLU()))

        d = enc_chans[-1]
        # Using no-bottleneck resblock
        for _ in range(num_resnet_blocks):
            enc_layers.append(ResBlock_1d_BN(d))
        enc_layers.append(nn.Conv1d(d, num_class, 1))

        # # Using bottleneck resblock
        # if num_resnet_blocks > 0:
        #     enc_layers.append(ResBlock_1d_bottleneck([d, d, d, 4 * d]))
        # for _ in range(num_resnet_blocks - 1):
        #     enc_layers.append(ResBlock_1d_bottleneck([4 * d, d, d, 4 * d]))
        # enc_layers.append(nn.Conv1d(4 * d, num_class, 1))

        self.encoder = nn.Sequential(*enc_layers)

        fc = [
              nn.ReLU(),
              nn.Conv1d(signal_len // (2 ** num_layers), fc_hdim, 1),
              nn.ReLU(),
              nn.Conv1d(fc_hdim, 1, 1),
              ]

        self.fc = nn.Sequential(*fc)

        self._register_external_parameters()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.input_key = input_key

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    def forward(
        self,
        x,
        return_loss = False,
        return_vec = True,
        temperature = 0.5,
        ):
        signal = x[self.input_key]
        device, num_class, signal_len = signal.device, self.num_class, self.signal_len

        assert signal.shape[-1] == signal_len, f'input must have the correct image size {signal_len}'
        
        signal = signal.view(-1, 1, self.signal_len)
        signal = self.encoder(signal)
        signal = signal.transpose(1,2)

        y_pred = self.fc(signal)[:, 0, :]

        if not return_loss:
            return y_pred

        y = x['cat']
        if y.dim() == 1:
            y = y[None, ...]
        loss = self.loss_fn(y_pred, y)
        # ### double check BCEWithLogitsLoss
        # _loss_fn = nn.BCELoss()
        # _loss = 0
        # for i in range(y_pred.shape[-1]):
        #     _loss += _loss_fn(torch.sigmoid(y_pred[:, i]), y[:, i])
        # assert _loss / y_pred.shape[-1] == loss
        if return_vec:
            return y_pred, loss
        else:
            return loss


class fmri_cat_linear(nn.Module):
    def __init__(
        self,
        signal_len = 512,
        num_class = 80,
        hdim1 = 256,
        hdim2 = 128,
        input_key = 'fmri',
        vec_key = 'cat', # output key
    ):
        super().__init__()

        self.signal_len = signal_len
        self.num_class = num_class

        fc = [
              nn.Linear(signal_len, hdim1),
              nn.ReLU(),
              nn.Linear(hdim1, hdim2),
              nn.ReLU(),
              nn.Linear(hdim2, num_class),
              ]

        self.fc = nn.Sequential(*fc)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.input_key = input_key
        self.vec_key = vec_key

    def forward(
        self,
        x,
        return_loss = False,
        return_vec = True,
        temperature = 0.5,
        ):
        signal = x[self.input_key]
        device, num_class, signal_len = signal.device, self.num_class, self.signal_len

        assert signal.shape[-1] == signal_len, f'input must have the correct image size {signal_len}'

        signal = signal.view(-1, 1, self.signal_len)

        y_pred = self.fc(signal)[:, 0, :]

        if not return_loss:
            return y_pred

        y = x[self.vec_key]
        if y.dim() == 1:
            y = y[None, ...]
        loss = self.loss_fn(y_pred, y)
        if return_vec:
            return y_pred, loss
        else:
            return loss


class clip_fmri_res(fmri_clip_res):
    def __init__(
        self,
        signal_len = 256,
        num_tokens = 512,
        num_layers = 1,
        num_resnet_blocks = 4,
        hidden_dim = 64,
        fc_hdim = 128,
        channels = 1,
        loss_type = 'mse_cos',
        cos_weight = 0.6,
        normalization = None,
        vec_key = 'clip_img',
    ):
        super().__init__(
            num_layers = 1,
            num_resnet_blocks = 0,
            loss_type = loss_type,
            cos_weight = cos_weight,
            normalization = normalization,
            vec_key = vec_key,
        )
        del self._modules['encoder']
        del self._modules['fc']
        # assert log2(signal_len).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.signal_len = signal_len
        self.num_tokens = num_tokens
        self.num_layers = num_layers

        hdim = hidden_dim
        dec_chans = list(reversed([hidden_dim] * num_layers))

        dec_init_chan = channels if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        dec_chans_io, = map(lambda t: list(zip(t[:-1], t[1:])), (dec_chans,))
        dec_layers = []

        for (dec_in, dec_out) in dec_chans_io:
            dec_layers.append(nn.Sequential(nn.ConvTranspose1d(dec_in, dec_out, 4, stride = 2, padding = 1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock_1d(dec_chans[1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv1d(channels, dec_chans[1], 1))

        dec_layers.append(nn.Conv1d(dec_chans[-1], channels, 1))

        self.decoder = nn.Sequential(*dec_layers)

        self.fc = nn.Sequential(
            nn.Conv1d(signal_len * (2 ** num_layers), fc_hdim, 1),
            nn.Conv1d(fc_hdim, num_tokens, 1),
        )

    def forward(
        self,
        x,
        return_loss = False,
        return_vec = True,
        temperature = 0.5,
        cos_weight_override = None,
    ):
        signal = x[self.vec_key]
        device, num_tokens, signal_len = signal.device, self.num_tokens, self.signal_len
        cos_weight = default(cos_weight_override, self.cos_weight)

        assert signal.shape[-1] == signal_len, f'input must have the correct image size {signal_len}'
        # signal: 512
        signal = signal.view(-1, 1, self.signal_len) # 1, 1, 512
        signal = self.norm(signal)

        signal = self.decoder(signal)
        signal = signal.transpose(1,2)
        y_pred = self.fc(signal)[..., 0]

        if not return_loss:
            return y_pred

        y = x['fmri']
        y = y[None, ...] if y.dim() == 1 else y

        if 'contrastive' in self.loss_type:
            if return_vec:
                return y, self.loss_fn(y_pred, y, temperature, cos_weight)
            return self.loss_fn(y_pred, y, temperature, cos_weight)
        elif self.loss_type == 'mse_cos':
            if return_vec:
                return y, self.loss_fn(y_pred, y, cos_weight)
            return self.loss_fn(y_pred, y, cos_weight)
        else: # mse, cos
            if return_vec:
                return y, self.loss_fn(y_pred, y)
            return self.loss_fn(y_pred, y)


if __name__ == '__main__':

    ########### CONFIGS ###########
    vec_key = 'cat'
    vec_key_allowed = ['clip_img', 'clip_cap', 'img_vec', 'cat'] # 'clip_img' for CLIP vectors and 'img_vec' for resnet vectors, 'cat' for object categories
    assert vec_key in vec_key_allowed, (
        f'vec_key {vec_key} is not in the allowed list {vec_key_allowed}')
    if vec_key[:4]=='clip':
        proj_name = 'fmri-clip'
    elif vec_key == 'img_vec':
        proj_name = 'fmri-dnn'
    elif vec_key == 'cat':
        proj_name = 'fmri-cat'

    use_wandb = True
    viz_data = True
    eval_mode = False
    loss_type = 'mse_cos' # 'cos' or 'contrastive' or 'mse' or 'mse_cos' or 'mse_cos_contrastive'
    use_vqvae = False # for 2fc model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wt_dir = rt_dir + 'data/weights/'
    model_name = wt_dir + 'fmri_cat_stuff_linear_subj05.pth'
    load_model = None
    netD_name = None#wt_dir + 'fmri_clipcapnorm_mse_cos_contra_thr_D.pth'
    load_netD = None

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    fmri_pad = 15744 # 13040 # 

    # # defaults for res_mse
    # defaults = dict(
    #     hidden_layer_size_vqvae=256,
    #     hidden_layer_size=3072,
    #     num_layers_res=2, # number of layers in the more complicated model
    #     num_resnet_blocks=4, # number of resnet blocks in the more complicated model
    #     fc_hdim=256, # for the more complicated model
    #     learn_rate=6.587e-4,
    #     wdecay=6.62e-6,
    #     epochs=20,
    #     )

    # defaults for res_mse_cos_aug
    defaults = dict(
        hidden_layer_size_vqvae=256,
        hidden_layer_size=3072,
        num_layers_res=1, # number of layers in the more complicated model
        num_resnet_blocks=4, # number of resnet blocks in the more complicated model
        fc_hdim=128, # for the more complicated model
        cos_weight=0.6, # weight of cosine similarity loss in mse+cos loss
        learn_rate=2.0e-4,
        # learn_rate=9.0e-4,
        base_lr=0.2,
        # wdecay=7.6e-6,
        wdecay=1e-5,
        # wdecay=1e-3,
        epochs=40,
        )

    # # defaults for VICReg
    # defaults = dict(
    #     num_layers_res=1, # number of layers in the more complicated model
    #     num_resnet_blocks=4, # number of resnet blocks in the more complicated model
    #     fc_hdim=128, # for the more complicated model
    #     hidden_layer_size=3072,
    #     projector_hidden_dim=2048,
    #     learn_rate=5.0e-5,
    #     base_lr=0.2,
    #     # wdecay=1.0e-3,
    #     wdecay=1.0e-6,
    #     epochs=50,
    #     )

    # # defaults for fmri-resnet
    # defaults = dict(
    #     hidden_layer_size_vqvae=256,
    #     hidden_layer_size=3072,
    #     num_layers_res=2, # number of layers in the more complicated model
    #     num_resnet_blocks=4, # number of resnet blocks in the more complicated model
    #     fc_hdim=256, # for the more complicated model
    #     cos_weight=0.8515, # weight of cosine similarity loss in mse+cos loss
    #     learn_rate=0.000226,
    #     base_lr=0.2,
    #     wdecay=0.0000747,
    #     epochs=40,
    #     )
    if use_wandb:
        wandb.init(project=proj_name, config=defaults, settings=wandb.Settings(start_method="fork"))
        wandb_config = wandb.config
        wandb.log({'model_name': model_name,})
    ########### CONFIGS DONE ###########

    if vec_key[:4] == 'clip':
        ''' fmri with CLIP image/caption vectors'''
        load_caption = True if (eval_mode or vec_key=='clip_cap') else False
        img_trans_p = 0 if (eval_mode or vec_key=='clip_cap') else 1.0#0.9
        # # use pre-computed CLIP vectors (no aug, slightly worse)
        # dataset = NSDwithCLIP(fmri_pad=fmri_pad, roi=ROI_VOX+'_zscored',
        #                            clip_norm=True, load_caption=load_caption,
        #                            threshold=1.5)

        # Use CLIP on the go with augmentations
        import CLIP.clip as clip
        clip_model = clip.load("ViT-B/32", device=device)
        dataset = NSDwithCLIP(fmri_pad=fmri_pad, roi=ROI_VOX+'_zscored',
                              CLIP=clip_model, img_trans=img_trans_p,
                              clip_norm=True, clip_std=False, clip_01=False,
                              load_caption=load_caption, threshold=1.5,#)
                              caption_selection='random',)
        vec_len = 512

    if vec_key == 'img_vec':
        ''' fmri & vicreg resnet embedding'''
        img_trans_p = 0
        chan_avg = True # False
        return_layer = 4 # 2
        dataset = NSDRes(fmri_pad=fmri_pad, roi=ROI_VOX+'_zscored',
                         img_trans=img_trans_p, img_size=128, return_layer=return_layer,
                         chan_avg=chan_avg, spatial_pool='max_mean',
                         vec_norm=False, vec_std=False, vec_01=True)
        vec_len = dataset.emb if chan_avg else len(dataset[0][vec_key])

    if vec_key == 'cat':
        dataset = NSDwithCLIP(load_fmri=True, fmri_pad=fmri_pad, roi=ROI_VOX+'_zscored',
                              load_img=True, img_trans=0, load_clip=False,
                              load_cat=True, cat_type='things_stuff')

        # def extra_fmri_process(x):
        #     x = torch.argmax(x, -2)
        #     num_fmri_tokens = 5000
        #     return x / float(num_fmri_tokens)

        # fmri_model = VQVAE_1d(fmri_pad, 5000, num_layers=4).to(device)
        # mask_ratio = 0
        # fmri_model_name = f'nsd_vqvae_984fmri5000_{mask_ratio}.pth'
        # load_trained_model(wt_dir + fmri_model_name, fmri_model)
        # fmri_model = fmri_model.dVAE

        # dataset = NSDwithCLIP(load_fmri=True, fmri_pad=fmri_pad, roi=ROI_VOX+'_zscored',
        #                       fmri_model=fmri_model, fmri_model_args={'return_logits':True},
        #                       extra_fmri_fn=extra_fmri_process,
        #                       load_img=True, img_trans=0, load_clip=False,
        #                       load_cat=True, cat_type='things_stuff')
        vec_len = dataset.num_class

    print(f'dataset length:{len(dataset)}, vector length {vec_len}.')
    fmri_len = dataset.get_fmri_shape()
    print(f'fmri length: {fmri_len}.')

    #################### Dataset sample visualization ####################
    if viz_data:
        sample = dataset[1]
        plt.plot(sample['fmri'].cpu().numpy())
        plt.show()

        if vec_key == 'cat':
            print(sample['cat'])
            print([dataset.cat_list[i] for i in list(torch.where(sample['cat'] == 1)[0])])

        else:
            plt.plot(sample[vec_key].cpu().numpy(), label='image')
            if vec_key[:4] == 'clip' and load_caption:
                plt.plot(sample['clip_cap'].cpu().numpy(), label='caption')
            plt.legend()
            plt.show()

            viz_len = 150
            repeats = 5
            # Same image different augs
            if img_trans_p > 0:
                for i in range(repeats):
                    sample = dataset[1]
                    plt.plot(sample[vec_key].cpu().numpy()[: viz_len], label=f'aug {i}')
                # plt.plot(sample['clip_cap'].cpu().numpy(), label='caption')
                plt.legend()
                plt.show()

            # Different images
            for i in range(repeats):
                sample = dataset[i]
                plt.plot(sample[vec_key].cpu().numpy()[: viz_len], label=f'img {i}')
            plt.legend()
            plt.show()

        # Heatmap
        if vec_key == 'img_vec' and not chan_avg:
            for i in range(repeats):
                sample = dataset.__getitem__(i, verbose=True)
                plt.imshow(sample['img'].cpu().numpy().transpose(1,2,0))
                plt.imshow(sample['img_vec_2d'].cpu().numpy(), alpha=0.5)
                plt.show()
    ################################################################################
    if 'contrastive' in loss_type:
        batch_size = 500 if eval_mode else 128 # VICReg one using 256 (128 with bottleneck resnet)
        print_every = 30
    else:
        batch_size = 500 if eval_mode else 32
        print_every = 100
    if use_wandb: wandb.log({'batch_size': batch_size,})
    train_set, val_set, train_loader, val_loader = get_dataloader(dataset, batch_size)

    if use_vqvae:
        model_dir = wt_dir + config['fmri_emb_model']['model_name']
        num_fmri_tokens = config['fmri_emb_data']['num_tokens']
        from fmri_vqvae import VQVAE_1d
        vqvae = VQVAE_1d(fmri_len=fmri_len, num_tokens=num_fmri_tokens, num_layers=4).to(device)
        # print(vqvae)
        load_trained_model(model_dir, vqvae, exact_load=True)
        vqvae = vqvae.dVAE
        start_dim = vqvae.signal_len // (2 ** vqvae.num_layers)
        hid_dim = wandb_config.hidden_layer_size_vqvae if use_wandb else (
            defaults['hidden_layer_size_vqvae'])

    else:
        vqvae = None
        start_dim = fmri_len
        hid_dim = wandb_config.hidden_layer_size if use_wandb else (
            defaults['hidden_layer_size'])

    # # the simpler model: 2FC layers
    # cos_weight = wandb_config.cos_weight if use_wandb else defaults['cos_weight']
    # model = fmri_clip_2fc(
    #     fmri_len = fmri_len,
    #     start_dim = start_dim,
    #     hid_dim = hid_dim,
    #     vqvae = vqvae,
    #     loss_type=loss_type,
    #     cos_weight = cos_weight,
    #     ).to(device)

    # the more complex model with resnet blocks
    num_layers_res = wandb_config.num_layers_res if use_wandb else (
        defaults['num_layers_res'])
    num_resnet_blocks = wandb_config.num_resnet_blocks if use_wandb else (
        defaults['num_resnet_blocks'])
    fc_hdim = wandb_config.fc_hdim if use_wandb else defaults['fc_hdim']
    cos_weight = wandb_config.cos_weight if use_wandb else defaults['cos_weight']

    if vec_key == 'cat':
        # model = fmri_cat(
        #     signal_len = fmri_len,
        #     num_class = dataset.num_class,
        #     num_layers = 1,
        #     num_resnet_blocks = 4,
        #     hidden_dim = 64,
        #     fc_hdim = 128,
        #     input_key = 'fmri',
        #     ).to(device)

        ### If from fMRI to cat
        hdim1 = 2048
        hdim2 = 512
        # ### If from VQVAE-fMRI to cat
        # hdim1 = 512
        # hdim2 = 256
        model = fmri_cat_linear(
            signal_len = fmri_len,
            num_class = dataset.num_class,
            hdim1 = hdim1,
            hdim2 = hdim2,
            input_key = 'fmri',
            ).to(device)
    else:
        model = fmri_clip_res(
            signal_len = fmri_len,
            num_tokens = vec_len,
            num_layers = num_layers_res,
            num_resnet_blocks = num_resnet_blocks,
            hidden_dim = 64,
            fc_hdim = fc_hdim,
            loss_type = loss_type,
            cos_weight = cos_weight,
            vec_key = vec_key,
            last_activation = False, # True if use vec_01 for dataset (res vec), else False (CLIP)
            ).to(device)

    # # VICReg model (with expander, and variance reg loss)
    # p_hdim = wandb_config.projector_hidden_dim if use_wandb else (
    #     defaults['p_hdim'])
    # num_layers_res = wandb_config.num_layers_res if use_wandb else (
    #     defaults['num_layers_res'])
    # num_resnet_blocks = wandb_config.num_resnet_blocks if use_wandb else (
    #     defaults['num_resnet_blocks'])
    # fc_hdim = wandb_config.fc_hdim if use_wandb else defaults['fc_hdim']
    # model = VICReg(
    #     signal_len = fmri_len,
    #     num_layers = num_layers_res,
    #     num_resnet_blocks = num_resnet_blocks,
    #     hidden_dim = 64,
    #     projector_hidden = (p_hdim, p_hdim, p_hdim),
    #     fc_hdim = fc_hdim,
    #     ).to(device)
    # load_trained_model(wt_dir + 'fmri_clipnorm_mse_cos_aug_thr.pth', model.backbone_net)

    print(model)
    try:
        load_trained_model(load_model, model)
    except:
        print(f'{load_model} NOT loaded.')

    # netD = Discriminator().to(device)
    # netD.apply(weights_init)
    # print(netD)
    # try:
    #     load_trained_model(load_netD, netD)
    # except:
    #     print(f'{load_netD} NOT loaded.')

    if eval_mode:
        # visualize
        val_idx = val_loader.batch_sampler.sampler.data_source.indices
        for i in range(5):
            idx = val_idx[i]
            viz_clip(idx, dataset, model, vec_key=vec_key, viz_len=200)

        # similarity evaluation
        x = next(iter(val_loader))
        sim = sim_eval(model, x, vec_key=vec_key, viz_idx=4)

    else:
        epochs = wandb_config.epochs if use_wandb else defaults['epochs']
        wdecay = wandb_config.wdecay if use_wandb else defaults['wdecay']
        learn_rate = wandb_config.learn_rate if use_wandb else defaults['learn_rate']

        # lars_optimizer = LARS(
        #     model.parameters(),
        #     lr=0,
        #     weight_decay=wdecay,
        #     weight_decay_filter=exclude_bias_and_norm,
        #     lars_adaptation_filter=exclude_bias_and_norm,
        # )
        # base_lr = wandb_config.base_lr if use_wandb else defaults['base_lr']

        # train_loss, val_loss = train_fmri_clip(
        train_loss, val_loss, train_errD, train_errG = train_fmri_clip(
            model, epochs, learn_rate, wdecay, None,
            train_loader, val_loader, vec_key=vec_key, print_every=print_every,
            # netD=netD,
            # optimizer=lars_optimizer, base_lr=base_lr, # if using LARS
            # lr_decay_epoch=8,
            save_model=model_name, save_all=True,
            # save_model=None,
            # save_netD=netD_name,
            # min_vloss=0.,
            # cos_weight_override=0.5,
            viz_loss=False, use_wandb=use_wandb)
        print(min(val_loss))
