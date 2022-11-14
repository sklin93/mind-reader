
import numpy as np
import torch
import torch.nn as nn
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
import torch.nn.functional as F
import torchvision.transforms as T
import clip
import dnnlib
import random
from training.networks import signed_max
# import matplotlib.pyplot as plt
# import CLIP.clip as clip
import sys
sys.path.append('/home/sikun/vicreg/')
from resnet import *
del sys.path[-1]


class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, real_features): # to be overridden by subclass
        raise NotImplementedError()

# trainable perturbations
class Model(torch.nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(512, 1024)
        self.linear2 = torch.nn.Linear(1024, 1024)
        self.linear3 = torch.nn.Linear(1024, 1024)
        self.linear4 = torch.nn.Linear(1024, 512)
        self.linear5 = torch.nn.Linear(512, 1024)
        self.linear6 = torch.nn.Linear(1024, 1024)
        self.linear7 = torch.nn.Linear(1024, 1024)
        self.linear8 = torch.nn.Linear(1024, 512)
        self.device = device

    def forward(self, x):
        mu = F.leaky_relu(self.linear1(x))
        mu = F.leaky_relu(self.linear2(mu))
        mu = F.leaky_relu(self.linear3(mu))
        mu = self.linear4(mu)
        std = F.leaky_relu(self.linear5(x))
        std = F.leaky_relu(self.linear6(std))
        std = F.leaky_relu(self.linear7(std))
        std = self.linear8(std)
        return mu + std.exp()*(torch.randn(mu.shape).to(self.device))

    def loss(self, real, fake, temp=0.1, lam=0.5):
        sim = torch.cosine_similarity(real.unsqueeze(1), fake.unsqueeze(0), dim=-1)
        if temp > 0.:
            sim = torch.exp(sim/temp)
            sim1 = torch.diagonal(F.softmax(sim, dim=1))*temp
            sim2 = torch.diagonal(F.softmax(sim, dim=0))*temp
            if 0.<lam < 1.:
                return -(lam*torch.log(sim1) + (1.-lam)*torch.log(sim2))
            elif lam == 0:
                return -torch.log(sim2)
            else:
                return -torch.log(sim1)
        else:
            return -torch.diagonal(sim)

#----------------------------------------------------------------------------

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_channels=3,
        zero_init_residual=False,
        groups=1,
        widen=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        last_activation="relu",
        ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(
            num_channels,
            num_out_filters,
            kernel_size=7,
            stride=2,
            padding=2,
            bias=False,
        )
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block,
            num_out_filters,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block,
            num_out_filters,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block,
            num_out_filters,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            last_activation=last_activation,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self, block, planes, blocks, stride=1, dilate=False,
        last_activation="relu"
        ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                last_activation=(last_activation if blocks == 1 else "relu"),
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    last_activation=(last_activation if i == blocks - 1 else "relu"),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, return_layer=4, chan_avg=False):
        assert return_layer in [1, 2, 3, 4], (
            'you can only return embeddings after layer 1, 2, 3, 4.')

        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 64, 56, 56
        x = self.layer1(x) # 256, 56, 56
        if return_layer == 1:
            return torch.flatten(self.avgpool(x), 1) if chan_avg else x
        x = self.layer2(x) # 512, 28, 28
        if return_layer == 2:
            return torch.flatten(self.avgpool(x), 1) if chan_avg else x
        x = self.layer3(x) # 1024, 14, 14
        if return_layer == 3:
            return torch.flatten(self.avgpool(x), 1) if chan_avg else x
        x = self.layer4(x) # 2048, 7, 7
        return torch.flatten(self.avgpool(x), 1) if chan_avg else x

def resnet50(layer=2, pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    emb = {1: 256, 2: 512, 3: 1024, 4: 2048}
    if pretrained:
        misc.load_trained_model('/home/sikun/vicreg/resnet50.pth', model)
    return model, emb[layer]

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, G_mani, D, augment_pipe=None, style_mixing_prob=0.9,
        r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, use_fmri=False, fmri_vec=None, fmri_vec2=None,
        resloss=False, ires=10, vec2_res=False): # vec2_dnn controls if the separate condition branch is using resnet vec
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.G_mani = G_mani
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        clip_model, _ = clip.load("ViT-B/32", device=device)  # Load CLIP model here
        self.clip_model = clip_model.eval()
        # self.mapper = Model(device)
        # self.mapper.load_state_dict(torch.load('./implicit.0.001.64.True.0.0.pth', map_location='cpu')) # path to the noise mapping network
        # self.mapper.to(device)
        self.use_fmri = use_fmri
        self.resloss = resloss
        self.ires = ires

        if use_fmri:
            assert fmri_vec is not None, 'must provide mapper model if use fmri for end to end traing'
            self.fmri_vec = fmri_vec
            if fmri_vec2 is not None:
                self.fmri_vec2 = fmri_vec2
                self.vec2_res = vec2_res
                if vec2_res:
                    self.return_layer = 2
                    resnet, _ = resnet50(layer=self.return_layer, zero_init_residual=True)
                    self.resnet = resnet.to(device).eval()
        if self.resloss:
            self.return_layer = 2
            resnet, _ = resnet50(layer=self.return_layer, zero_init_residual=True)
            self.resnet = resnet.to(device).eval()

    def run_G(self, z, c, sync, txt_fts=None, ):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)

            if self.style_mixing_prob > 0:
                new_ws = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)

                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = new_ws[:, cutoff:]

        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws, fts=txt_fts)
        return img, ws

    def run_D(self, img, c, sync, fts=None, structure=2):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        if structure == 4:
            with misc.ddp_sync(self.D, sync):
                logits, d_fts, d_fts2 = self.D(img, c, fts=fts)
            return logits, d_fts, d_fts2
        else:
            with misc.ddp_sync(self.D, sync):
                logits, d_fts = self.D(img, c, fts=fts)
            return logits, d_fts

    def run_res(self, x):
        _x = self.full_preprocess(x, ratio=1.0, fix_size=128)
        img_vec = self.resnet(_x, chan_avg=False, return_layer=self.return_layer)
        # with torch.no_grad():
            # img_vec = self.resnet(_x, chan_avg=False, return_layer=self.return_layer)
        max_vec = img_vec.max(1)[0]
        mean_vec = img_vec.mean(1)
        img_vec = max_vec / max_vec.max() + 2 * mean_vec / mean_vec.max()
        return img_vec.flatten(start_dim=1)

    def normalize(self):
        return T.Compose([
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def full_preprocess(self, img, mode='bicubic', ratio=0.5, fix_size=224):
        full_size = img.shape[-2]

        if full_size < fix_size:
            pad_1 = torch.randint(0, fix_size-full_size, ())
            pad_2 = torch.randint(0, fix_size-full_size, ())
            m = torch.nn.ConstantPad2d((pad_1, fix_size-full_size-pad_1, pad_2, fix_size-full_size-pad_2), 1.)
            reshaped_img = m(img)
        else:
            if ratio == 1:
                cropped_img = img
            else:
                cut_size = torch.randint(int(ratio*full_size), full_size, ())
                left = torch.randint(0, full_size-cut_size, ())
                top = torch.randint(0, full_size-cut_size, ())
                cropped_img = img[:, :, top:top+cut_size, left:left+cut_size]
            reshaped_img = F.interpolate(cropped_img, (fix_size, fix_size), mode=mode, align_corners=False)
        # print(f'reshaped_img range before {reshaped_img.min()}, {reshaped_img.max()}')
        reshaped_img = (reshaped_img + 1.)*0.5 # range in [0., 1.] now
        # print(f'reshaped_img range after {reshaped_img.min()}, {reshaped_img.max()}')
        # TODO double check this for structure==4 after training
        reshaped_img = self.normalize()(reshaped_img)
        return  reshaped_img

    def custom_preprocess(self, img, ind, cut_num, mode='bicubic'):   # more to be implemented here
        full_size = img.shape[-2]

        grid = np.sqrt(cut_num)
        most_right = min(int((ind%grid + 1)*full_size/grid), full_size)
        most_bottom = min(int((ind//grid + 1)*full_size/grid), full_size)

        cut_size = torch.randint(int(full_size//(grid+1)), int(min(min(full_size//2, most_right), most_bottom)), ()) # TODO: tune this later
        left = torch.randint(0, most_right-cut_size, ())
        top = torch.randint(0, most_bottom-cut_size, ())
        cropped_img = img[:, :, top:top+cut_size, left:left+cut_size]
        reshaped_img = F.interpolate(cropped_img, (224, 224), mode=mode, align_corners=False)

        reshaped_img = (reshaped_img + 1.)*0.5 # range in [0., 1.] now
        reshaped_img = self.normalize()(reshaped_img)

        return  reshaped_img

    def contra_loss(self, temp, mat1, mat2, lam):
        sim = torch.cosine_similarity(mat1.unsqueeze(1), mat2.unsqueeze(0), dim=-1)
        if temp > 0.:
            # sim = torch.exp(sim/temp) # TODO: This implementation is incorrect, it should be sim=sim/temp. change hp
            sim = sim/temp
            # However, this incorrect implementation can reproduce our results with provided hyper-parameters.
            # If you want to use the correct implementation, please manually revise it.
            # The correct implementation should lead to better results, but don't use our provided hyper-parameters, you need to carefully tune lam, temp, itd, itc and other hyper-parameters
            sim1 = torch.diagonal(F.softmax(sim, dim=1))*temp
            sim2 = torch.diagonal(F.softmax(sim, dim=0))*temp
            if 0.<lam < 1.:
                return lam*torch.log(sim1) + (1.-lam)*torch.log(sim2)
            elif lam == 0:
                return torch.log(sim2)
            else:
                return torch.log(sim1)
        else:
            return torch.diagonal(sim)

    def mse_cos_contra_loss(self, temp, y_pred, y, lam, a1=0.2, a2=0.3, a3=0.5):
        l_mse = nn.MSELoss()(y_pred, y)
        target = torch.ones(len(y)).to(y.device)
        l_cos = nn.CosineEmbeddingLoss()(y_pred, y, target)
        return a1*l_mse + a2*l_cos - a3*self.contra_loss(temp, y_pred, y, lam).mean()

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, img_fts, lam, temp,
        gather, d_use_fts, itd, itc, iid, iic, mixing_prob=0., txt_fts=None, fmri=None, structure=2,
        f_dim=512, f_dim2=512):
        # print(phase)
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'Mmain', 'Mboth', 'Mreg', 'M2main', 'M2both', 'M2reg'], phase

        if self.use_fmri:
            txt_fts_gt = txt_fts.clone()
            assert fmri is not None, 'fmri data must be provided if using fmri'
            # get txt_fts from mapper model
            txt_fts = self.fmri_vec(fmri)
            # print(f'txt_fts through mapper: {txt_fts.shape}')
            if structure == 4:
                # assert self.fmri_vec2 is not None
                txt_fts_2 = self.fmri_vec2(fmri)
                if self.vec2_res:
                    txt_fts_2 = txt_fts_2 - 0.5 # make it [-0.5, 0.5 range]

                    # plt.plot(txt_fts[0].detach().cpu().numpy(), label='before norm fts')
                    # plt.plot(txt_fts_2[0].detach().cpu().numpy(), label='before norm fts2')
                    # plt.legend()
                    # plt.show()

                    # Normalize to make it the same scale as txt_fts
                    txt_fts = txt_fts/txt_fts.norm(dim=-1, keepdim=True)
                    txt_fts_2 = txt_fts_2/txt_fts_2.norm(dim=-1, keepdim=True)
                    # plt.plot(txt_fts[0].detach().cpu().numpy())
                    # plt.plot(txt_fts_2[0].detach().cpu().numpy())
                    # plt.show()

                # txt_fts = torch.cat((txt_fts, txt_fts_2), -1)
                # txt_fts_gt2 = None
            if structure == 5:
                txt_fts_2 = self.fmri_vec2(fmri)
                txt_fts = signed_max(txt_fts, txt_fts_2)

        else:
            # Use_fmri==False case the conditions are concatenated, separate them here
            if self.structure == 4 and len(txt_fts) > f_dim:
                txt_fts_2 = txt_fts[:, f_dim : f_dim + f_dim2]
                # txt_fts_gt2 = txt_fts[:, f_dim+f_dim2:]
                txt_fts = txt_fts[:, :f_dim]

        # assert txt_fts is not None
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Mmain = (phase in ['Mmain', 'Mboth'])
        do_M2main = (phase in ['M2main', 'M2both'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
        # do_Mreg  = (phase in ['Mreg', 'Mboth']) # TODO
        # do_M2reg  = (phase in ['M2reg', 'M2both'])

        # augmentation
        aug_level_1 = 0.1
        aug_level_2 = 0.75
        # print(torch.cosine_similarity(img_fts, txt_fts, dim=-1))

        # the semantic similarity of perturbed feature with real feature would be:
        # sim >= (sqrt(1 - aug_level^2)-aug_level)/(sqrt(1 + 2*aug_level*sqrt(1 - aug_level^2)))
        # mixing_prob = mixing_prob # probability to use img_fts instead of txt_fts
        random_noise = torch.randn(txt_fts.shape).to(img_fts.device)# + torch.randn((1, 512)).to(img_fts.device)
        random_noise = random_noise/random_noise.norm(dim=-1, keepdim=True)
        txt_fts_ = txt_fts*(1-aug_level_1) + random_noise*aug_level_1

        # print(txt_fts_.shape, txt_fts_2.shape)#, txt_fts_gt2.shape, txt_fts_gt2.shape[-1])
        txt_fts_ = txt_fts_/txt_fts_.norm(dim=-1, keepdim=True)

        if structure == 4:
            random_noise = torch.randn(txt_fts_2.shape).to(img_fts.device)# + torch.randn((1, 512)).to(img_fts.device)
            random_noise = random_noise/random_noise.norm(dim=-1, keepdim=True)
            txt_fts_2 = txt_fts_2*(1-aug_level_1) + random_noise*aug_level_1
            txt_fts_2 = txt_fts_2/txt_fts_2.norm(dim=-1, keepdim=True)
            # if txt_fts_gt2 is not None and txt_fts_gt2.shape[-1] > 0:
                # txt_fts_gt2 = txt_fts_gt2/txt_fts_gt2.norm(dim=-1, keepdim=True)
            txt_fts_1 = txt_fts_ # IT IS PASS BY REF, TAKE CARE
            txt_fts_ = torch.cat((txt_fts_1, txt_fts_2), -1)
            # txt_fts_.requires_grad_()
        #     print(phase, txt_fts_1.requires_grad, txt_fts_2.requires_grad, txt_fts_.requires_grad)
        # else:
        #     print(phase, txt_fts_.requires_grad)

        # print(txt_fts_1.shape, txt_fts_2.shape, txt_fts_.shape)
        # plt.plot(txt_fts_1[0].detach().cpu().numpy(), label='img')
        # plt.plot(txt_fts_2[0].detach().cpu().numpy(), label='cap')
        # plt.legend()
        # plt.show()

        if txt_fts.shape[-1] == img_fts.shape[-1]:
            # Gaussian purterbation
            img_fts_ = img_fts*(1-aug_level_2) + random_noise*aug_level_2

            # learned generation
            # with torch.no_grad():
            #     normed_real_full_img = self.full_preprocess(real_img, ratio=0.99)
            #     img_fts_real_full_ = self.clip_model.encode_image(normed_real_full_img).float()
            #     img_fts_real_full_ = img_fts_real_full_/img_fts_real_full_.norm(dim=-1, keepdim=True)
            #     # img_fts_real_full_ = img_fts
            #     img_fts_ = self.mapper(img_fts_real_full_) + img_fts_real_full_

            img_fts_ = img_fts_/img_fts_.norm(dim=-1, keepdim=True)
            if mixing_prob > 0.99:
                txt_fts_ = img_fts_
            elif mixing_prob < 0.01:
                txt_fts_ = txt_fts_
            else:
                txt_fts_ = torch.where(torch.rand([txt_fts_.shape[0], 1], device=txt_fts_.device) < mixing_prob, img_fts_, txt_fts_)

        img_img_d = iid if structure != 4 else iid / 2. # discriminator
        img_img_c = iic   # clip
        img_txt_d = itd if structure != 4 else itd / 2.  # discriminator
        img_txt_c = itc if structure != 4 else itc / 2.  # clip
        temp = temp
        lam = lam

        def gather_tensor(input_tensor, gather_or_not):
            if gather_or_not:
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                output_tensor = [torch.zeros_like(input_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(output_tensor, input_tensor)
                output_tensor[rank] = input_tensor
              # print(torch.cat(output_tensor).size())
                return torch.cat(output_tensor)
            else:
                return input_tensor
        
        if structure == 4:
            txt_fts_all = gather_tensor(txt_fts_1, gather)
            txt_fts_2_all = gather_tensor(txt_fts_2, gather)
            # if txt_fts_gt2 is not None and txt_fts_gt2.shape[-1] > 0:
                # txt_fts_gt2_all = gather_tensor(txt_fts_gt2, gather)
        else:
            txt_fts_all = gather_tensor(txt_fts_, gather)
            # print('txt_fts_all:', txt_fts_all.shape)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _ = self.run_G(gen_z, gen_c, txt_fts=txt_fts_, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                if structure == 4:
                    gen_logits, gen_d_fts, gen_d_fts_2 = self.run_D(gen_img, gen_c, sync=False, fts=txt_fts_, structure=structure)
                else:
                    gen_logits, gen_d_fts = self.run_D(gen_img, gen_c, sync=False, fts=txt_fts_, structure=structure)

                gen_d_fts_all = gather_tensor(gen_d_fts, gather)
                if structure == 4:
                    gen_d_fts_2_all = gather_tensor(gen_d_fts_2, gather)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))

                # print(f'gen img shape: {gen_img.shape}, range: {gen_img.min(), gen_img.max()}')
                normed_gen_full_img = self.full_preprocess(gen_img)
                # print(f'normed gen img shape: {normed_gen_full_img.shape}, range: {normed_gen_full_img.min(), normed_gen_full_img.max()}')
                img_fts_gen_full = self.clip_model.encode_image(normed_gen_full_img)
                img_fts_gen_full = img_fts_gen_full/img_fts_gen_full.norm(dim=-1, keepdim=True)

                img_fts_gen_full_all = gather_tensor(img_fts_gen_full, gather)
                img_fts_all = gather_tensor(img_fts, gather)
                if img_txt_c > 0.:
                    clip_loss_img_txt = self.contra_loss(temp, img_fts_gen_full_all, txt_fts_all, lam)
                    loss_Gmain = loss_Gmain - img_txt_c*clip_loss_img_txt.mean()
                    if structure == 4:
                        # If using different model like resnet, use contra_loss(temp, resnet(normed_gen_full_img), txt_fts_2_all, lam)
                        if self.vec2_res:
                            img_vec = self.run_res(gen_img)
                            # to 0 - 1
                            img_vec -= 0.67
                            img_vec /= 2.33
                            # to -0.5 - 0.5
                            img_vec -= 0.5
                            img_vec = img_vec/img_vec.norm(dim=-1, keepdim=True)
                            img_vec_all = gather_tensor(img_vec, gather)
                            # plt.plot(img_vec_all[0].detach().cpu().numpy(), label='from gen img')
                            # plt.plot(txt_fts_2_all[0].detach().cpu().numpy(), label='txt')
                            # plt.legend()
                            # plt.show()
                            vec_loss_img_txt = self.contra_loss(temp, img_vec_all, txt_fts_2_all, lam)
                            loss_Gmain = loss_Gmain - img_txt_c*vec_loss_img_txt.mean()
                        else:
                            clip_loss_img_txt2 = self.contra_loss(temp, img_fts_gen_full_all, txt_fts_2_all, lam)
                            loss_Gmain = loss_Gmain - img_txt_c*clip_loss_img_txt2.mean()
                            # Can also use contra_loss(temp, clip_txt(cap(gen_img)), txt_fts_2_all, lam), but this requires caption model

                if img_img_c > 0.: # using ground truth one
                    clip_loss_img_img = self.contra_loss(temp, img_fts_gen_full_all, img_fts_all, lam)
                    loss_Gmain = loss_Gmain - img_img_c*clip_loss_img_img.mean()
                    # similarly, contra_loss(temp, clip_txt(cap(gen_img)), txt_fts_gt2_all, lam), but this requires caption model

                if img_txt_d > 0.:
                    loss_Gmain = loss_Gmain - img_txt_d*self.contra_loss(temp, gen_d_fts_all, txt_fts_all, lam).mean()
                    if structure == 4:
                        loss_Gmain = loss_Gmain - img_txt_d*self.contra_loss(temp, gen_d_fts_2_all, txt_fts_2_all, lam).mean()

                if img_img_d > 0.:
                    if structure == 4:
                        with torch.no_grad():
                            _, g_real_d_fts, g_real_d_fts_2 = self.run_D(real_img.detach(), real_c, sync=False, fts=txt_fts_, structure=structure)
                        g_real_d_fts_all = gather_tensor(g_real_d_fts, gather)
                        g_real_d_fts_2_all = gather_tensor(g_real_d_fts_2, gather)
                        loss_Gmain = loss_Gmain - img_img_d*self.contra_loss(temp, g_real_d_fts_all, gen_d_fts_all, lam).mean()
                        loss_Gmain = loss_Gmain - img_img_d*self.contra_loss(temp, g_real_d_fts_2_all, gen_d_fts_2_all, lam).mean()
                    else:
                        with torch.no_grad():
                            _, g_real_d_fts = self.run_D(real_img.detach(), real_c, sync=False, fts=txt_fts_, structure=structure)
                        g_real_d_fts_all = gather_tensor(g_real_d_fts, gather)
                        loss_Gmain = loss_Gmain - img_img_d*self.contra_loss(temp, g_real_d_fts_all, gen_d_fts_all, lam).mean()

                if self.resloss:
                    img_vec_gen = self.run_res(gen_img)
                    img_vec_real = self.run_res(real_img.detach())
                    img_vec_gen_all = gather_tensor(img_vec_gen, gather)
                    img_vec_real_all = gather_tensor(img_vec_real, gather)
                    # plt.plot(img_vec_real_all[0].detach().cpu().numpy(), label='real')
                    # plt.plot(img_vec_gen_all[0].detach().cpu().numpy(), label='gen')
                    # plt.legend()
                    # plt.show()
                    loss_Gmain = loss_Gmain - self.ires * self.contra_loss(temp, img_vec_real_all, img_vec_gen_all, lam).mean()

                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink

                if structure == 4:
                    txt_fts_1_0 = txt_fts_1[:batch_size]
                    txt_fts_2_0 = txt_fts_2[:batch_size]
                    txt_fts_0 = torch.cat((txt_fts_1_0,txt_fts_2_0), -1)
                else:
                    txt_fts_0 = txt_fts_[:batch_size]
                txt_fts_0.requires_grad_()
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], txt_fts=txt_fts_0, sync=sync)

                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    if d_use_fts:
                        pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws, txt_fts_0], create_graph=True, only_inputs=True)[0]
                    else:
                         pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _ = self.run_G(gen_z, gen_c, txt_fts=txt_fts_, sync=False)
                rt = self.run_D(gen_img, gen_c, sync=False, fts=txt_fts_, structure=structure) # Gets synced by loss_Dreal.
                gen_logits = rt[0]

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                if structure == 4:
                    real_logits, real_d_fts, real_d_fts_2 = self.run_D(real_img_tmp, real_c, sync=sync, fts=txt_fts_, structure=structure)
                else:
                    real_logits, real_d_fts = self.run_D(real_img_tmp, real_c, sync=sync, fts=txt_fts_, structure=structure)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    if img_txt_d > 0.:
                        real_d_fts_all = gather_tensor(real_d_fts, gather)
                        loss_Dreal = loss_Dreal - img_txt_d*self.contra_loss(temp, real_d_fts_all, txt_fts_all, lam).mean()
                        if structure == 4:
                            real_d_fts_2_all = gather_tensor(real_d_fts_2, gather)
                            loss_Dreal = loss_Dreal - img_txt_d*self.contra_loss(temp, real_d_fts_2_all, txt_fts_2_all, lam).mean()

                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        if do_Mmain or do_M2main: # must under use_fmri condition (set in training_loop)
            gen_img, _ = self.run_G(gen_z, gen_c, txt_fts=txt_fts_, sync=(sync and not do_Gpl)) # May get synced by Gpl.
            rt = self.run_D(gen_img, gen_c, sync=False, fts=txt_fts_, structure=structure)
            loss_GD = 0.1 * torch.nn.functional.softplus(-rt[0]) # -log(sigmoid(gen_logits))

            if do_Mmain:
                with torch.autograd.profiler.record_function('Mmain_forward'):
                    # first branch's gt is clip_img
                    img_fts_all = gather_tensor(img_fts, gather)
                    loss_M = self.mse_cos_contra_loss(temp, txt_fts_all, img_fts_all, lam)
                    loss_M = loss_M + loss_GD.mean()

                with torch.autograd.profiler.record_function('Mmain_backward'):
                    loss_M.mul(gain).backward()

            if do_M2main:
                with torch.autograd.profiler.record_function('M2main_forward'):
                    # second branch's gt is clip_cap
                    txt_fts_gt_all = gather_tensor(txt_fts_gt, gather)
                    loss_M2 = self.mse_cos_contra_loss(temp, txt_fts_all, txt_fts_gt_all, lam)
                    loss_M2 = loss_M2 + loss_GD.mean()

                with torch.autograd.profiler.record_function('M2main_backward'):
                    loss_M2.mul(gain).backward()
