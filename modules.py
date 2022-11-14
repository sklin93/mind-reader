""" Mostly unused, but experimented modules """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from nsd_utils import *
from dalle_pytorch.dalle_pytorch import *


class ResBlock_1d(nn.Module):
    # 1d discrete vae
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(chan, chan, 3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class DiscreteVAE(nn.Module):
    def __init__(
        self,
        signal_len = 256,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        num_resnet_blocks = 0,
        hidden_dim = 64,
        channels = 3,
        smooth_l1_loss = False,
        temperature = 0.9,
        straight_through = False,
        kl_div_loss_weight = 0.,
        normalization = None,
    ):
        super().__init__()
        # assert log2(signal_len).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.signal_len = signal_len
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv1d(enc_in, enc_out, 4, stride = 2, padding = 1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose1d(dec_in, dec_out, 4, stride = 2, padding = 1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock_1d(dec_chans[1]))
            enc_layers.append(ResBlock_1d(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv1d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv1d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv1d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        # take care of normalization within class
        self.normalization = normalization

        self._register_external_parameters()

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

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, signals):
        logits = self(signals, return_logits = True)
        codebook_indices = logits.argmax(dim = 1).flatten(1)
        return codebook_indices

    def decode(
        self,
        sig_seq
    ):
        signal_embeds = self.codebook(sig_seq)
        b, n, d = signal_embeds.shape
        # h = w = int(sqrt(n))

        # image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        signal_embeds = torch.transpose(signal_embeds, 1, 2) # b d n
        images = self.decoder(signal_embeds)
        return images

    def forward(
        self,
        signal,
        return_loss = False,
        return_recons = False,
        return_logits = False,
        temp = None
    ):
        device, num_tokens, signal_len, kl_div_loss_weight = signal.device, self.num_tokens, self.signal_len, self.kl_div_loss_weight
        assert signal.shape[-1] == signal_len, f'input must have the correct image size {signal_len}'
        signal = self.norm(signal)
        
        logits = self.encoder(signal)

        if return_logits:
            return logits # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        # ipdb.set_trace()
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through)
        # sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        sampled = einsum('b n l, n d -> b d l', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss
        recon_loss = self.loss_fn(signal, out)

        # kl divergence

        # logits = rearrange(logits, 'b n h w -> b (h w) n')
        logits = torch.transpose(logits, 1, 2)
        log_qy = F.log_softmax(logits, dim = -1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out


class VQVAE_1d(torch.nn.Module):
    def __init__(self, fmri_len, num_tokens, num_layers=5, num_resnet_blocks=3, hidden_dim=64):
        '''
        num_layers = 5,           # nsdgeneral --> 492
        num_layers = 4,           # for bold5k & nsd visual cortex, 3392 --> 212; nsd visual cortex --> 830; nsdgeneral --> 984 (4 layer)
        num_layers = 6,           # for bold5k all voxels (downsamples 2 ** num_layers) --> 57600 --> 900; nsdgeneral --> 246
        '''
        super().__init__()
        self.dVAE = DiscreteVAE(
            signal_len = fmri_len,
            num_layers = num_layers,
            num_tokens = num_tokens,    
            codebook_dim = 1024,       # codebook dimension
            hidden_dim = hidden_dim,          # hidden dimension
            channels = 1,
            num_resnet_blocks = num_resnet_blocks,    # number of resnet blocks
            temperature = 0.9,        # gumbel softmax temperature, the lower this is, the harder the discretization
            straight_through = False, # straight-through for gumbel softmax. unclear if it is better one way or the other
            normalization = None,     #((0.49539,), (0.13273,)),
            )
        self.fmri_len = fmri_len

    def forward(self, x):
        x = x.view(-1, 1, self.fmri_len)
        return self.dVAE(x).flatten(1)


class ResBlock_1d_BN(nn.Module):
    ''' with batchnorm, wth bottleneck formulation, with last activation'''
    def __init__(self, chan, first_kernel=3):
        ''' - chan: a length 4 list'''
        super().__init__()

        if type(chan).__name__[:3] == 'int':
            chan = [chan] * 4
        assert len(chan) == 4, 'need to provide 4 channel numbers for 3 layers'

        first_pad = (first_kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(chan[0], chan[1], first_kernel, padding=first_pad),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(chan[1]),
            nn.Conv1d(chan[1], chan[2], 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(chan[2]),
            nn.Conv1d(chan[2], chan[3], 1),
            # nn.BatchNorm1d(chan[3]),
        )
        self.last_activation = nn.ReLU(inplace=True)

        self.upsample = None
        if chan[0] != chan[3]:
            self.upsample = nn.Sequential(
                nn.Conv1d(chan[0], chan[3], 1),
                # nn.BatchNorm1d(chan[3])
                )

    def forward(self, x):
        if self.upsample is None:
            return self.last_activation(self.net(x) + x)
        else:
            return self.last_activation(self.net(x) + self.upsample(x))


class ResBlock_1d_bottleneck(ResBlock_1d_BN):
    def __init__(self, chan):
        ResBlock_1d_BN.__init__(self, chan, first_kernel=1)


class MLP(nn.Module):
    ''' From https://github.com/FloCF/SSL_pytorch/blob/main/models/utils.py'''
    def __init__(self, in_dim: int,
                 hidden_dims: Union[int, tuple],
                 bias: bool = True,
                 use_batchnorm: bool = True,
                 batchnorm_last: bool = False):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)

        mlp = [nn.Linear(in_dim, hidden_dims[0], bias = bias)]
        for i in range(len(hidden_dims) - 1):
            if use_batchnorm:
                mlp.append(nn.BatchNorm1d(hidden_dims[i]))
            bias = False if (i + 2 == len(hidden_dims)) else bias
            mlp.extend([nn.ReLU(inplace=True),
                        nn.Linear(hidden_dims[i], hidden_dims[i+1], bias=bias)])
        if batchnorm_last:
            # for simplicity, remove gamma in last BN
            mlp.append(nn.BatchNorm1d(hidden_dims[-1], affine=False))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


class VICReg(nn.Module):
    """
    Code adapted from https://github.com/FloCF/SSL_pytorch/blob/main/models/vicreg.py
    """
    def __init__(self, signal_len, num_layers=1, num_resnet_blocks=4,
                 hidden_dim=64, fc_hdim=128, vec_key='clip_img',
                 projector_hidden: Union[int, tuple] = (2048,2048,2048),
                 λ: float = 25., μ: float = 25., ν: float = 1.,
                 γ: float = 1., ϵ: float = 1e-4):
        super().__init__()
        # Loss Hyperparams
        self.lambd = λ
        self.mu = μ
        self.nu = ν
        self.gamma = γ
        self.eps = ϵ

        self.backbone_net = fmri_clip_res(
            signal_len = signal_len,
            num_layers = num_layers,
            num_resnet_blocks = num_resnet_blocks,
            hidden_dim = hidden_dim,
            fc_hdim = fc_hdim,
            last_activation = False,
            )

        self.repre_dim = 512
        self.projector = MLP(self.repre_dim, projector_hidden)
        self.vec_key = vec_key

    def loss_fn(self, z1, z2, λ, μ, ν, γ, ϵ):
        # Get batch size and dim of rep
        assert z1.shape == z2.shape, (
            f'z1 shape {z1.shape} not equal to z2 shape {z2.shape}')
        N, D = z1.shape

        # invariance loss
        sim_loss = F.mse_loss(z1, z2)

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        # variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + ϵ)
        std_z2 = torch.sqrt(z2.var(dim=0) + ϵ)
        std_loss = (torch.mean(F.relu(γ - std_z1)) / 2 +
                    torch.mean(F.relu(γ - std_z2)) / 2)

        # covariance loss
        cov_z1 = (z1.T @ z1) / (N - 1)
        cov_z2 = (z2.T @ z2) / (N - 1)
        cov_loss = (off_diagonal(cov_z1).pow_(2).sum().div(D)+
                    off_diagonal(cov_z2).pow_(2).sum().div(D))

        return λ*sim_loss + μ*std_loss + ν*cov_loss

    def forward(self, x, return_loss=False):
        fmri_vec = self.backbone_net(x)

        if not return_loss:
            return fmri_vec

        z1 = self.projector(fmri_vec)
        z2 = self.projector(x[self.vec_key])

        return self.loss_fn(z1, z2,
                            self.lambd, self.mu, self.nu, self.gamma, self.eps)


class Discriminator(nn.Module):
  def __init__(self, vec_dim=512, hdim1=256, hdim2=128):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(vec_dim, hdim1), 
      nn.LeakyReLU(0.25),
      nn.Linear(hdim1, hdim2),
      nn.LeakyReLU(0.25),
      nn.Linear(hdim2, 1),
    #   nn.Sigmoid()
    )

  def forward(self, x):
    """Forward pass"""
    return self.layers(x)
