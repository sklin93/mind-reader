import os
import sys
import h5py
import json
import yaml
import pickle
import numpy as np
import nibabel as nib
import scipy.io as spio
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import zscore

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from dalle_pytorch import DiscreteVAE
import CLIP.clip as clip
vicreg_path = '/home/sikun/vicreg/'
sys.path.append(vicreg_path)
from resnet import *

with open('nsd_config.yaml') as f:
    nsd_config = yaml.load(f, Loader=yaml.Loader)
    # print(nsd_config)

SUBJ = '01'
DATA_DIR = nsd_config['data']['data_dir']
voxel_size = nsd_config['data']['voxel_size']
ROI_FILES = nsd_config['data']['roi_files']
ROI_VOX = nsd_config['data']['roi_vox']

STIM_FILE = os.path.join(DATA_DIR, 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
STIM_ORDER_FILE = os.path.join(DATA_DIR, 'nsddata/experiments/nsd/nsd_expdesign.mat')
STIM_INFO = os.path.join(DATA_DIR, 'nsddata/experiments/nsd/nsd_stim_info_merged.pkl')
STIM_CAP = os.path.join(DATA_DIR, f'nsddata_stimuli/stimuli/nsd/annotations/nsd_captions_{SUBJ}.json')
# os.path.join(DATA_DIR, 'nsddata_stimuli/stimuli/nsd/annotations/', f'nsd_captions_vetted_{SUBJ}.pkl')
STIM_CAT = os.path.join(DATA_DIR, f'nsddata_stimuli/stimuli/nsd/annotations/nsd_cat_{SUBJ}.json')
STIM_CAT_STUFF = os.path.join(DATA_DIR, f'nsddata_stimuli/stimuli/nsd/annotations/nsd_cat_stuff_{SUBJ}.json')
STIM_STUFF = os.path.join(DATA_DIR, f'nsddata_stimuli/stimuli/nsd/annotations/nsd_stuff_{SUBJ}.json')

FMRI_DIR = os.path.join(DATA_DIR, f'nsddata_betas/ppdata/subj{SUBJ}/func'+voxel_size+'/betas_fithrf_GLMdenoise_RR/')
ROI_FILES = [os.path.join(DATA_DIR, f'nsddata/ppdata/subj{SUBJ}/func'+voxel_size+'/roi/',
                          ROI_FILE) for ROI_FILE in ROI_FILES]
ROI_VOX = os.path.join(DATA_DIR, f'nsddata_betas/ppdata/subj{SUBJ}/func'+voxel_size+'/', ROI_VOX)

TRIAL_PER_SESS = 750
SESS_NUM = 37
MAX_IDX = TRIAL_PER_SESS * SESS_NUM
device = torch.device(nsd_config['device'])


def load_trained_model(model_name, model, exact_load=True):
    '''usage: load_trained_model(data_dir + model_name, model)'''
    pretrained_dict = torch.load(model_name)
    model_dict = model.state_dict()
    if not exact_load:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in
                           model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Model weight loaded from {}.'.format(model_name))

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def thresholding(vec, thr):
    ''' thresholding vec values to [-thr, thr]. '''
    if thr > 0:
        _thr = torch.ones_like(vec) * thr
        vec = torch.minimum(vec, _thr)
        vec = torch.maximum(vec, -_thr)
    return vec

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def extract_voxels(fmri_dir, roi_files, out_dir, regions=None, flatten=False):
    ''' Extract voxels defined by roi_file.
    Served as a preprocessing to save time during sample loading.
    Write extracted vectors into hdf5 files under output_dir.

    - fmri_dir: where the fmri betas are located.
                Each beta file should have shape (750, 83, 104, 81).
    - roi_files: a list of files defining ROI. All have shape (81, 104, 83).
                 Voxels values out of interest are -1.
                 The else ranges from 0 to the (#region + 1).
    - regions: a dict. {str(roi_file_name): list(interested regions)}
               e.g. regions = {'prf-visualrois': [2, 3]}
               If provided, the funciton will only care about the voxels
               in the given regions. If None, will care about all voxels not -1.
    - flatten: if True, return a flattened vector for each fmri sample.
               (only voxels in the input roi_file are kept)
    - out_dir: where to write the new fmri to.
    '''
    assert os.path.isdir(out_dir), f'mkdir for the output directory {out_dir} first!'

    mask = []
    for roi_file in roi_files:
        roi_name = os.path.basename(roi_file)[:-7]
        _mask = nib.load(roi_file).get_fdata()
        available_region = [int(r) for r in set(_mask.flatten())]
        print(f'Extracting ROI based on {roi_name},',
              f'available_regions: {available_region}')
        available_region.remove(-1)
        region_count = [_mask == a_r for a_r in available_region]
        region_count = [np.count_nonzero(r_c) for r_c in region_count]
        print(region_count)

        if regions and roi_name in regions:
            for r in regions[roi_name]:
                assert r in available_region, (
                    f'region index {r} is not available for ROI {roi_name}!')
            _mask = np.stack([_mask == r for r in regions[roi_name]]).sum(0)
            _mask = (_mask != 0)
            print(f'Region {regions[roi_name]} voxel count:',
                  np.count_nonzero(_mask))
        else:
            _mask = (_mask > 0)
        print(f'ROI voxel count: {np.count_nonzero(_mask)}')
        mask.append(_mask)

    mask = np.stack(mask).sum(0)
    mask = (mask != 0)
    # mask axis orders are different from fmri axis order, do transpose
    mask = mask.T
    print(f'\nTotal ROI voxel count: {np.count_nonzero(mask)}\n', flush=True)

    # processing all fmri data
    fmri_files = [f for f in os.listdir(fmri_dir) if
                  os.path.isfile(os.path.join(fmri_dir, f)) and
                  f[-5:] == '.hdf5']
    for fmri_file in fmri_files:
        print(os.path.join(fmri_dir, fmri_file), flush=True)
        with h5py.File(os.path.join(fmri_dir, fmri_file), 'r') as f:
            fmri = f['betas'][()]
        if flatten:
            fmri = [fmri[trial][mask] for trial in range(len(fmri))]
            fmri = np.stack(fmri)
        else:
            fmri = fmri * mask[None,...]
        # save new fmri
        out_f = os.path.join(out_dir, fmri_file)
        with h5py.File(out_f, 'w') as f:
            dset = f.create_dataset('betas', data=fmri)

def get_dataloader(dataset, batch_size=32):
    ''' Give a whole dataset, seperate train set and val set so that they have different images'''
    stim_order = loadmat(STIM_ORDER_FILE)
    ''' Go through all samples to build a dict with keys being their stimulus (image) IDs. '''
    sig = {}
    # for idx in range(MAX_IDX):
    for idx in range(len(dataset)):
        ''' nsdId as in design csv files'''
        nsdId = stim_order['subjectim'][int(SUBJ)-1, stim_order['masterordering'][idx] - 1] # - 1
        if nsdId not in sig:
            sig[nsdId] = []
        sig[nsdId].append(idx)
    print(len(sig.keys()))

    ''' prepare dataloader '''
    train_idx_len = int(len(sig.keys()) * 0.85)
    train_idx = list(sig.keys())[: train_idx_len]
    val_idx = list(sig.keys())[train_idx_len:]

    train_idx = sorted(np.concatenate([sig[idx] for idx in train_idx]))
    val_idx = sorted(np.concatenate([sig[idx] for idx in val_idx]))
    print(f'num training samples {len(train_idx)}, val samples {len(val_idx)}')

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size)

    # Sanity check dataloader
    try:
        next(iter(train_loader))
    except:
        print('Cant load, double check if lengths for different samples are same, None type, etc.')
    print(f'train loader iter: {len(train_loader)}, val loader iter: {len(val_loader)}')
    return train_set, val_set, train_loader, val_loader


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class RandomMask(object):
    ''' Apply random cutouts (masking) on an image'''
    def __init__(self, mask_ratio=0.5):
        ''' - mask_ratio: the ratio of (cutout area width or height) / (image width or height)'''
        self.cx = np.random.rand()
        self.cy = np.random.rand()
        self.m = mask_ratio / 2.0

    def __call__(self, sample):
        cx, cy, m = self.cx, self.cy, self.m
        _, x, y = sample.shape

        start_x = round((cx - m) * x)
        start_y = round((cy - m) * y)
        end_x = round((cx + m) * x)
        end_y = round((cy + m) * y)

        mask = torch.ones_like(sample)
        mask[:, max(0, start_x): min(x-1, end_x), max(0, start_y): min(y-1, end_y)] = 0

        return sample * mask


def get_img_trans(extra_aug=0.9, toPIL=True, img_size=256, color_jitter_p=0.4,
                  gray_scale_p=0.2, gaussian_blur_p=0.5, masking_p=1.0,
                  masking_ratio=0.3):
    '''
    - extra_aug: a value between 0-1. If 0, only apply resizing and to tensor.
                 If > 0, this p controls the probability that an augmentation
                 is actually implemented.
    - toPIL: bool. CLIP need PIL to process, if using other models, set it to F.
    - color_jitter_p: ADA 0.4, VICReg 0.8
    - gray_scale_p: VICReg 0.2
    - gaussian_blur_p: VICReg0.5, similar to ADA's filter.
                       (ADA has multiple, and has p = 1.0)
    - masking_ratio: ADA 0.5
    '''

    img_trans = []
    img_trans.append(transforms.ToTensor())
    img_trans.append(transforms.Resize((img_size, img_size)))

    run_extra = np.random.rand()
    if bool(extra_aug) and (run_extra < extra_aug):
        img_trans.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))
        img_trans.append(transforms.RandomHorizontalFlip(p=0.5))

        cj = np.random.rand()
        # print(f'color jitter {cj}, {cj < color_jitter_p}')
        if cj < color_jitter_p:
            img_trans.append(transforms.ColorJitter(0.4, 0.4, 0.2, 0.1))
        gs = np.random.rand()
        # print(f'grayscale {gs}, {gs < gray_scale_p}')
        if gs < gray_scale_p:
            img_trans.append(transforms.Grayscale(num_output_channels=3))
        gb = np.random.rand()
        # print(f'gaussian blur {gb}, {gb < gaussian_blur_p}')
        if gb < gaussian_blur_p:
            img_trans.append(transforms.GaussianBlur(kernel_size=23))
        # img_trans.append(transforms.RandomSolarize(128, p=0.1))

        img_trans.append(RandomMask(masking_ratio))

    if toPIL:
        img_trans.append(transforms.ToPILImage())
    img_trans = transforms.Compose(img_trans)
    return img_trans

def get_mask(roi, subj, hem_ext='', offset=10, verbose=True):
    ''' Get corresponding roi index on the nsdgeneral vector,
        used for masking nsdgeneral vector inputs.
        - roi: string, f'v{i}' or f'floc-{xxx}'
        - subj: string, f'0{i}'
        - hem_ext: choose from '', 'lh.', 'rh.'
    '''
    if roi == '':
        assert len(hem_ext) > 0, 'provide either ROI or hemisphere choice'
        with open(f'/home/sikun/NSD/data/pos_map_hem_subj{subj}.pkl', 'rb') as f:
            mask_idx = pickle.load(f)[hem_ext]
    elif roi in [f'v{i+1}' for i in range(4)]: # V1-4, for checking hierarchy
        with open(f'/home/sikun/NSD/data/{hem_ext}pos_map_v1234_subj{subj}.pkl', 'rb') as f:
            mask_idx = pickle.load(f)[roi]
    else: # for checking task specific ROIs
        with open(f'/home/sikun/NSD/data/{hem_ext}pos_map_detailed_subj{subj}.pkl', 'rb') as f:
            dict_data = pickle.load(f)
        mask_idx = []
        for k, v in dict_data.items():
            if k[:len(roi)] == roi:
                mask_idx += v
    mask_idx = [i + offset for i in mask_idx] # account for fMRI padding

    if verbose:
        print(roi, hem_ext, len(mask_idx), mask_idx[:10])

    return mask_idx


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
        load_trained_model(vicreg_path + 'resnet50.pth', model)
    return model, emb[layer]

def _load_cat_init(cat_type):
    ''' helper func for loading categories (used in dataset init) '''
    cat_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis','snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner','blanket','branch','bridge','building-other','bush','cabinet','cage','cardboard','carpet','ceiling-other','ceiling-tile','cloth','clothes','clouds','counter','cupboard','curtain','desk-stuff','dirt','door-stuff','fence','floor-marble','floor-other','floor-stone','floor-tile','floor-wood','flower','fog','food-other','fruit','furniture-other','grass','gravel','ground-other','hill','house','leaves','light','mat','metal','mirror-stuff','moss','mountain','mud','napkin','net','paper','pavement','pillow','plant-other','plastic','platform','playingfield','railing','railroad','river','road','rock','roof','rug','salad','sand','sea','shelf','sky-other','skyscraper','snow','solid-other','stairs','stone','straw','structural-other','table','tent','textile-other','towel','tree','vegetable','wall-brick','wall-concrete','wall-other','wall-panel','wall-stone','wall-tile','wall-wood','water-other','waterdrops','window-blind','window-other','wood']
    with open(STIM_INFO, 'rb') as f:
        stim_info = pickle.load(f, encoding='latin1')
    if cat_type == 'things':
        cat_f = STIM_CAT
        cat_list = cat_list[:80]
    elif cat_type == 'stuff':
        cat_f = STIM_STUFF
        cat_list = cat_list[80:]
    elif cat_type == 'things_stuff':
        cat_f = STIM_CAT_STUFF
    with open(cat_f, 'r') as f:
        nsd_cat = json.load(f)
    num_class = len(cat_list)
    print('number of classes:', num_class)

    return cat_list, stim_info, nsd_cat, num_class

def _load_fmri_init(roi=None, fmri_pad=None, fmri_model=None):
    fmri_dir = roi if roi else FMRI_DIR
    fmri_files = [f for f in os.listdir(fmri_dir) if
                  os.path.isfile(os.path.join(fmri_dir, f)) and
                  f[-5:] == '.hdf5']

    if fmri_pad:
        with h5py.File(os.path.join(fmri_dir, fmri_files[0]),
                       'r') as f:
            x = f['betas'][0]
            num_voxel = x.shape[-1]
        left_pad = (fmri_pad - num_voxel) // 2
        right_pad = fmri_pad - num_voxel - left_pad
    else:
        left_pad = right_pad = 0

    if fmri_model is not None:
        fmri_model.to(device)
        fmri_model.eval()

    return fmri_dir, fmri_files, left_pad, right_pad, fmri_model

def _load_fmri_forward(idx, fmri_dir, fmri_files, fmri_pad, left_pad, right_pad,
                       fmri_model=None, extra_fmri_fn=None, fmri_model_args={},
                       verbose=False):
    ''' helper func for loading fMRI (used in dataset forward) '''
    sess = idx // TRIAL_PER_SESS + 1
    fmri_file = os.path.join(fmri_dir, f'{fmri_files[0][:-7]}{sess:02}.hdf5')
    with h5py.File(fmri_file, 'r') as f:
        fmri_sample = f['betas'][idx % TRIAL_PER_SESS]
    if verbose:
        print('fmri loaded from', fmri_file)
        print('fmri shape:', fmri_sample.shape)
        print('beta min, mean, max:', fmri_sample.min(),
            fmri_sample.mean(), fmri_sample.max())

    fmri_sample = torch.FloatTensor(fmri_sample).to(device)
    if fmri_pad:
        fmri_sample = F.pad(fmri_sample, (left_pad, right_pad),
                            'constant', 0)
    if fmri_model is not None:
        with torch.no_grad():
            fmri_sample = fmri_model(fmri_sample.unsqueeze(0), **fmri_model_args)
    if extra_fmri_fn is not None:
        fmri_sample = extra_fmri_fn(fmri_sample)
    return fmri_sample

class NSDDataset(Dataset):
    def __init__(self, pt=False, load_img=False, img_trans=None,
                 load_fmri=False, fmri_pad=None, roi=None, load_caption=False,
                 load_cat=False, caption_selection='first', tokenizer=None,
                 text_pad=None):
        '''
        Support loading one or more of: image, fmri betas, text captions/categories.
        - pt: if the returned sample will be a pytorch tensor or not.
        - load_img, load_fmri, load_caption, load_cat: all bool,
          choose the modalities you need.
        - roi: string, the directory contains extracted ROI voxels.
               if None, return the 3d fmri activity: (83, 104, 81) for 1.8mm
        - caption_selection: either 'first' or 'all' or 'random'. Each image has
                            multiple captions. 'first' will only keep the first,
                            and 'all' will concatenate all captions, 'random'
                            will randomly choose one.

        if pt == True (aka using PyTorch), the following args/flags can be used:
        - img_trans: torchvision Transformations, optional.
        - fmri_pad: int, pad fMRI vector to this length. Useful when model
                    requires inputs to have a certain shape.
        - tokenizer: to convert text into tokens, tokenizer classes defined in 
                     https://shorturl.at/brxA4.
        - text_pad: int, pad tokenized text sequence into a fixed length.
        '''
        assert load_img or load_fmri or load_caption or load_caption, (
            'You must choose to load at least one modeality!'
            )
        
        if load_img or load_caption or load_cat:
            stim_order = loadmat(STIM_ORDER_FILE)
            self.subjectim = stim_order['subjectim']
            self.masterordering = stim_order['masterordering']
            if load_caption or load_cat:
                with open(STIM_INFO, 'rb') as f:
                    self.stim_info = pickle.load(f, encoding='latin1')
                if load_caption:
                    with open(STIM_CAP, 'r') as f:
                        self.nsd_captions = json.load(f)
                if load_cat:
                    with open(STIM_CAT, 'r') as f:
                        self.nsd_cat = json.load(f)                
       
        if load_img:
            self.stim_file = STIM_FILE
            if pt and (img_trans is None):
                img_trans = transforms.ToTensor()
            self.img_trans = img_trans
       
        if load_fmri:
            self.fmri_dir = roi if roi else FMRI_DIR
            self.fmri_files = [f for f in os.listdir(self.fmri_dir) if
                               os.path.isfile(os.path.join(self.fmri_dir, f)) and
                               f[-5:] == '.hdf5']
        
        self.pt = pt
        self.load_img = load_img
        self.load_fmri = load_fmri
        self.fmri_pad = fmri_pad
        self.load_caption = load_caption
        self.load_cat = load_cat
        assert caption_selection in ['first', 'random', 'all'], (
            "you must choose from 'first', 'random', 'all' as your caption selection method")
        self.caption_selection = caption_selection
        self.tokenizer = tokenizer
        self.text_pad = text_pad

    def __len__(self):
        return TRIAL_PER_SESS * len(self.fmri_files)

    def _get_caption(self, cap, method):
        if method == 'first':
            cap = cap[0]
        elif method == 'random':
            rand_id = np.random.choice(len(cap))
            cap = cap[rand_id]
        else:
            cap = ' '.join(cap)
        return cap

    def get_fmri_shape(self):
        with h5py.File(os.path.join(self.fmri_dir,
                                    self.fmri_files[0]), 'r') as f:
            s = f['betas'][0].shape
            print(f'shape: {s}')
            num_vox = f['betas'][0].flatten().shape[0]
            print(f'num voxel: {num_vox}')
            if self.fmri_pad:
                print(f'padded to: {self.fmri_pad}')
                return self.fmri_pad
            return num_vox

    def __getitem__(self, idx, verbose=False):
        sample = {}
        multi = False if type(idx).__name__[:3] == 'int' else True
        if self.load_img or self.load_caption or self.load_cat:
            nsdId = self.subjectim[int(SUBJ)-1, self.masterordering[idx] - 1] - 1
            if verbose:
                print(f'stim nsd id: {nsdId}, aka #{nsdId+1} in the tsv files.')

        if self.load_img:
            # dealing with multiple indexing,
            # to ensure the index maintains increasing order.
            if multi:
                nsdId_order = np.argsort(nsdId)
                _map = {nsdId_order[i] : i for i in range(len(nsdId_order))}
                nsdId_order = [_map[i] for i in range(len(nsdId_order))]
            with h5py.File(self.stim_file, 'r') as f:
                _nsdId = np.sort(nsdId) if multi else nsdId
                img_sample = f['imgBrick'][_nsdId]
            img_sample = img_sample[nsdId_order] if multi else img_sample
            if verbose:
                print('stim shape:', img_sample.shape)                        
            if self.pt:
                if self.img_trans:
                    img_sample = self.img_trans(img_sample)
                img_sample = img_sample.to(device)
            sample['img'] = img_sample

        if self.load_fmri:
            idx_array = np.arange(default(idx.start, 0),
                                  default(idx.stop, self.__len__()), idx.step
                                  ) if multi else (idx)
            sess = idx_array // TRIAL_PER_SESS + 1
            if multi:
                fmri_sample = []
                for s in range(len(sess)):
                    fmri_file = os.path.join(
                        self.fmri_dir,
                        f'{self.fmri_files[0][:-7]}{sess[s]:02}.hdf5')
                    with h5py.File(fmri_file, 'r') as f:
                        fmri_sample.append(f['betas']
                                           [idx_array[s] % TRIAL_PER_SESS])
                fmri_sample = np.stack(fmri_sample)
            else:
                fmri_file = os.path.join(
                    self.fmri_dir, f'{self.fmri_files[0][:-7]}{sess:02}.hdf5')
                with h5py.File(fmri_file, 'r') as f:
                    fmri_sample = f['betas'][idx_array % TRIAL_PER_SESS]                
                if verbose:
                    print('fmri loaded from', fmri_file)
                    print('fmri shape:', fmri_sample.shape)
                    print('beta min, mean, max:', fmri_sample.min(),
                        fmri_sample.mean(), fmri_sample.max())
            # # standardize
            # fmri_sample = zscore(fmri_sample)
            if self.pt:
                # fmri_sample = torch.FloatTensor(fmri_sample).to(device)
                fmri_sample = torch.Tensor(fmri_sample).to(device)
                # if fmri_sample.ndim == 3: # 3d voxel volumn
                #     fmri_sample = fmri_sample.flatten()
                # if fmri_sample.ndim == 4: # batch, 3d volumn
                #     fmri_sample = fmri_sample.flatten(start_dim=1)
                if self.fmri_pad:
                    left_pad = (self.fmri_pad - fmri_sample.shape[-1]) // 2
                    right_pad = self.fmri_pad - fmri_sample.shape[-1] - left_pad
                    fmri_sample = F.pad(fmri_sample, (left_pad, right_pad),
                                        'constant', 0)
            sample['fmri'] = fmri_sample

        if self.load_caption:
            if multi:
                caption = []
                for id in nsdId:
                    cap = self.nsd_captions[str(self.stim_info['cocoId'][id])]
                    caption.append(self._get_caption(cap, self.caption_selection))
            else:
                caption = self.nsd_captions[str(self.stim_info['cocoId']
                                                [nsdId])]
                if verbose:
                    print('Captions:')
                    for a in caption:
                        print(a)
                caption = [self._get_caption(caption, self.caption_selection)]

            if self.pt and self.tokenizer:
                caption = [torch.tensor(self.tokenizer.encode(cap)).to(device)
                           for cap in caption]
                caption = torch.stack(caption).squeeze()
                if self.text_pad:
                    caption = F.pad(caption,
                                    (0, self.text_pad - caption.shape[-1]),
                                    'constant', 0)
            sample['caption'] = caption

        if self.load_cat:
            if multi:
                cat = []
                for id in nsdId:
                    cat.append(self.nsd_cat[str(self.stim_info['cocoId'][id])])
            else:
                cat = [self.nsd_cat[str(self.stim_info['cocoId'][nsdId])]]
                if verbose:
                    print('Categories:', cat)
            sample['cat'] = cat
        return sample


class NSD_v(Dataset):
    ''' each sample contains sample['v123'] and sample['v4'] that correspind to
        voxel activities on V1-V3, and V4.
    '''
    def __init__(self, nsd_config, fmri_pad=None, bucketize=None):
        '''
        - fmri_pad: a dictionary {'v123': padded vetor length for v123,
                                  'v4': padded vector length for v4}
        - bucketize: None if return original signal. Or a int represent number
                     of tokens used (e.g. 10000)
        '''
        voxel_size = nsd_config['data']['voxel_size']
        self.fmri_v123_dir = os.path.join(nsd_config['data']['data_dir'],
                                          f'nsddata_betas/ppdata/subj{SUBJ}/func{voxel_size}/prf-visualrois-v123_zscored/')
        self.fmri_v4_dir = os.path.join(nsd_config['data']['data_dir'],
                                        f'nsddata_betas/ppdata/subj{SUBJ}/func{voxel_size}/prf-visualrois-v4_zscored/')
        self.fmri_v123_files = [f for f in os.listdir(self.fmri_v123_dir) if
                                os.path.isfile(os.path.join(self.fmri_v123_dir, f)) and
                                f[-5:] == '.hdf5']
        self.fmri_v4_files = [f for f in os.listdir(self.fmri_v4_dir) if
                              os.path.isfile(os.path.join(self.fmri_v4_dir, f)) and
                              f[-5:] == '.hdf5']
        self.trial_per_session = 750
        self.fmri_pad = fmri_pad # default is 837 * (2**7)
        self.bucketize = bucketize

    def __len__(self):
        len_v123 = len(self.fmri_v123_files)
        len_v4 = len(self.fmri_v4_files)
        assert len_v123 == len_v4, (
            f'number of v123 files {len_v123} not equal to number of v4 files {len_v4}.')
        return self.trial_per_session * len_v123

    def get_fmri_shape(self):
        ret = []
        with h5py.File(os.path.join(self.fmri_v123_dir,
                                    self.fmri_v123_files[0]), 'r') as f:
            num_vox_v123 = f['betas'][0].flatten().shape[0]
            print(f'num voxel in V1 V2 V3: {num_vox_v123}')
            if self.fmri_pad and 'v123' in self.fmri_pad:
                print(f"padded to: {self.fmri_pad['v123']}")
                ret.append(self.fmri_pad['v123'])
            else:
                ret.append(num_vox_v123)
        with h5py.File(os.path.join(self.fmri_v4_dir,
                                    self.fmri_v4_files[0]), 'r') as f:
            num_vox_v4 = f['betas'][0].flatten().shape[0]
            print(f'num voxel in V4: {num_vox_v4}')
            if self.fmri_pad and 'v4' in self.fmri_pad:
                print(f"padded to: {self.fmri_pad['v4']}")
                ret.append(self.fmri_pad['v4'])
            else:
                ret.append(num_vox_v4)
        return ret

    def _bucketizing(self, fmri_sample):
        fmri_sample -= fmri_sample.min()
        fmri_sample /= fmri_sample.max()
        fmri_sample = np.ceil(fmri_sample * self.bucketize)
        fmri_sample = np.minimum(fmri_sample, self.bucketize - 1)
        fmri_sample = torch.LongTensor(fmri_sample).to(device)
        return fmri_sample, fmri_sample.float().mean().item()

    def _process_fmri(self, idx, fmri_file, bucketize=False, fmri_pad=None):
        with h5py.File(fmri_file, 'r') as f:
            fmri_sample = f['betas'][idx % self.trial_per_session]
        # standardize
        fmri_sample = zscore(fmri_sample)

        # for CE loss when using Transformers
        if bucketize:
            fmri_sample, pad_value = self._bucketizing(fmri_sample)
        else:
            fmri_sample = torch.FloatTensor(fmri_sample).to(device)
            pad_value = 0

        if fmri_pad:
            left_pad = (fmri_pad - fmri_sample.shape[-1]) // 2
            right_pad = fmri_pad - fmri_sample.shape[-1] - left_pad
            fmri_sample = F.pad(fmri_sample, (left_pad, right_pad),
                                'constant', pad_value)
        return fmri_sample

    def __getitem__(self, idx):
        sample = {}
        sess = idx // self.trial_per_session + 1

        fmri_v123_file = os.path.join(
            self.fmri_v123_dir, f'{self.fmri_v123_files[0][:-7]}{sess:02}.hdf5')
        fmri_v4_file = os.path.join(
            self.fmri_v4_dir, f'{self.fmri_v4_files[0][:-7]}{sess:02}.hdf5')
        v123_pad = self.fmri_pad['v123'] if (
            self.fmri_pad and 'v123' in self.fmri_pad) else None
        v4_pad = self.fmri_pad['v4'] if (
            self.fmri_pad and 'v4' in self.fmri_pad) else None

        sample['v123'] = self._process_fmri(idx, fmri_v123_file,
                                            fmri_pad=v123_pad)
        sample['v4'] = self._process_fmri(idx, fmri_v4_file,
                                          bucketize=self.bucketize,
                                          fmri_pad=v4_pad)

        return sample


class NSDfmri(Dataset):
    def __init__(self, nsd_config, roi=None, fmri_pad=15744, bucketize=False,
                 num_tokens=10000):
        '''
        - roi: string, the directory contains extracted ROI voxels.
               if None, return the 3d fmri activity: (83, 104, 81) for 1.8mm
        '''
        voxel_size = nsd_config['data']['voxel_size']
        FMRI_DIR = os.path.join(nsd_config['data']['data_dir'], 
                                f'nsddata_betas/ppdata/subj{SUBJ}/func{voxel_size}/betas_fithrf_GLMdenoise_RR/')
        self.fmri_dir = roi if roi else FMRI_DIR
        self.fmri_files = [f for f in os.listdir(self.fmri_dir) if
                           os.path.isfile(os.path.join(self.fmri_dir, f)) and
                           f[-5:] == '.hdf5']
        with h5py.File(os.path.join(self.fmri_dir,
                                    self.fmri_files[0]), 'r') as f:
            self.trial_per_session = f['betas'].shape[0]
        self.fmri_pad = fmri_pad # default is 837 * (2**7)
        self.bucketize = bucketize
        self.num_tokens = num_tokens

    def __len__(self):
        return self.trial_per_session * len(self.fmri_files)

    def get_fmri_shape(self):
        with h5py.File(os.path.join(self.fmri_dir,
                                    self.fmri_files[0]), 'r') as f:
            num_vox = f['betas'][0].flatten().shape[0]
            print(f'num voxel: {num_vox}')
            if self.fmri_pad:
                print(f'padded to: {self.fmri_pad}')
                return self.fmri_pad
            return num_vox

    def __getitem__(self, idx):
        sample = {}
        sess = idx // self.trial_per_session + 1

        fmri_file = os.path.join(
            self.fmri_dir, f'{self.fmri_files[0][:-7]}{sess:02}.hdf5')
        with h5py.File(fmri_file, 'r') as f:
            fmri_sample = f['betas'][idx % self.trial_per_session]

        if self.bucketize:
            # normalize to 0 - 1
            fmri_sample -= fmri_sample.min()
            fmri_sample /= fmri_sample.max()
            fmri_sample = np.ceil(fmri_sample * self.num_tokens)
            fmri_sample = np.minimum(fmri_sample, self.num_tokens - 1)
            fmri_sample = torch.LongTensor(fmri_sample).to(device)
        else:
            # # standardize
            # fmri_sample = zscore(fmri_sample)
            fmri_sample = torch.FloatTensor(fmri_sample).to(device)
        if self.fmri_pad:
            left_pad = (self.fmri_pad - fmri_sample.shape[-1]) // 2
            right_pad = self.fmri_pad - fmri_sample.shape[-1] - left_pad
            fmri_sample = F.pad(fmri_sample, (left_pad, right_pad),
                                'constant', 0)
        sample['fmri'] = fmri_sample

        return sample


class NSDRes(Dataset):
    def __init__(self, load_fmri=True, fmri_pad=None, roi=None,
                 fmri_model=None, fmri_model_args={}, extra_fmri_fn=None,
                 load_vec=True, dnn=None, emb=0, return_layer=None, chan_avg=False,
                 spatial_pool='max_mean', vec_norm=False, vec_std=False, vec_01=False,
                 img_trans=0.9, img_size=128, threshold=0.0,
                 load_cat=False, cat_type='things_stuff',
                 load_vec_mapped=False, mapper=None,
                 ):
        '''
        - fmri_pad: int, pad fMRI vector to this length. Useful when model
                    requires inputs to have a certain shape.        
        - roi: string, the directory contains extracted ROI voxels.
               if None, return the 3d fmri activity: (83, 104, 81) for 1.8mm
        - dnn: image encoding model. By default, using VICReg resnet50 backbone,
               output after layer2 (length512).
        - emb: int, the img emb vector length.

        - return_layer: which layer of dnn should be returned. If None, return
                        the final dnn model output.
        - chan_avg: the returned vector is channel-wise averaged (vec has length
                    num_channel). If False, will take max_pool / mean_pool over
                    channels at each spatial location.
        - spatial_pool: if chan_avg if False, this will control the spatial
                        pooling method. Choose from 'max', 'mean', 'max_mean'.

        - img_trans: a value between 0-1. If 0, only apply resizing and to tensor.
                     If > 0, this p controls the probability that an augmentation
                     images will be augmented with probability p before passed to dnn.
        - threshold: to thresholding image embedding vector values, set to <= 0
                     if no thresholding is wanted.
        - vec_norm: bool. If True, the img emb vector will be normalized to a
                    unit sphere.
        - vec_std: bool. If True, the img emb vector will be standardized.
        - vec_01: bool. If True, the img emb vector will be normalized to 0-1.
        '''
        self.load_fmri = load_fmri
        self.load_cat = load_cat
        self.load_vec = load_vec
        self.load_vec_mapped = load_vec_mapped

        stim_order = loadmat(STIM_ORDER_FILE)
        self.subjectim = stim_order['subjectim']
        self.masterordering = stim_order['masterordering']
        self.stim_file = STIM_FILE

        if load_fmri or load_vec_mapped:
            self.fmri_dir, self.fmri_files, self.left_pad, self.right_pad, self.fmri_model = (
                _load_fmri_init(roi=roi, fmri_pad=fmri_pad, fmri_model=fmri_model))
            self.fmri_model_args = fmri_model_args
            self.fmri_pad = fmri_pad
            self.extra_fmri_fn = extra_fmri_fn

            if load_vec_mapped:
                assert mapper is not None
                self.mapper = mapper.to(device)
                self.mapper.eval()

        if load_vec:
            if dnn is None:
                self.dnn, self.emb = resnet50(layer=return_layer, zero_init_residual=True)
                self.dnn = self.dnn.to(device)
            else:
                assert emb > 0, 'provide correct image embedding vector length.'
                self.dnn = dnn.to(device)
                self.emb = emb
            self.dnn.eval()

            self.return_layer = return_layer
            self.chan_avg = chan_avg
            allowed_spatial_pool = ['max', 'mean', 'max_mean']
            assert spatial_pool in allowed_spatial_pool, (
                f'{spatial_pool} not supported, choose from {allowed_spatial_pool}.'
            )
            self.spatial_pool = spatial_pool
            self.img_trans = img_trans
            self.img_size = img_size

            assert (vec_norm and vec_std and vec_01) is False, (
                'normalization or standardization cannot be applied together.')
            self.vec_norm = vec_norm
            self.vec_std = vec_std
            self.vec_01 = vec_01
            self.threshold = threshold

        if load_cat:
            self.cat_list, self.stim_info, self.nsd_cat, self.num_class = _load_cat_init(cat_type)

    def __len__(self):
        return TRIAL_PER_SESS * len(self.fmri_files)

    def _scale_(self, vec):
        ''' 
        Without thresholding:
        min, max of res vecs are: 0, 18.4030
        mean, std of res vecs are: 3.3524, 3.1316
        '''
        if self.spatial_pool == 'max_mean':
            layer_min = {1: 0.44, 2: 0.67, 3: 1.0, 4: 0.19}
            layer_max = {1: 3.0, 2: 3.0, 3: 3.0, 4: 3.0}
            layer_mean = {1: 1.6537, 2: 1.785, 3: 2.1, 4: 1.3934}
            layer_std = {1: 0.2, 2: 0.181, 3: 0.2135, 4: 0.682}
        else:
            raise NotImplementedError

        if self.vec_norm:
            vec = vec / torch.norm(vec, p=2, dim=-1)
        elif self.vec_std:
            # TODO: not the correct way to do, but...
            vec_mean = vec.mean(-1) if self.threshold else layer_mean[self.return_layer]
            vec_std = vec.std(-1) if self.threshold else layer_std[self.return_layer]
            vec = (vec - vec_mean) / vec_std
        elif self.vec_01:
            vec_min =  0.0 if self.chan_avg else layer_min[self.return_layer]
            if self.threshold:
                vec_max = self.threshold
            else:
                vec_max = 18.5 if self.chan_avg else layer_max[self.return_layer]
            vec -= vec_min
            vec /= (vec_max - vec_min)
        return vec

    def get_fmri_shape(self):
        with h5py.File(os.path.join(self.fmri_dir,
                                    self.fmri_files[0]), 'r') as f:
            s = f['betas'][0].shape
            print(f'shape: {s}')
            num_vox = f['betas'][0].flatten().shape[0]
            print(f'num voxel: {num_vox}')
            if self.fmri_pad:
                print(f'padded to: {self.fmri_pad}')
                return self.fmri_pad
            return num_vox

    def __getitem__(self, idx, verbose=False):
        sample = {}
        nsdId = self.subjectim[int(SUBJ)-1, self.masterordering[idx] - 1] - 1
        sample['nsdId'] = nsdId

        if verbose:
            print(f'stim nsd id: {nsdId}, aka #{nsdId+1} in the tsv files.')

        ##### image & dnn emb #####
        if self.load_vec:
            with h5py.File(self.stim_file, 'r') as f:
                _image = f['imgBrick'][nsdId]
                _image = get_img_trans(extra_aug=self.img_trans, toPIL=False,
                                       img_size=self.img_size)(_image).to(device)
                if verbose: sample['img'] = _image

            with torch.no_grad():
                if self.return_layer:
                    img_vec = self.dnn(_image.unsqueeze(0), chan_avg=self.chan_avg,
                                       return_layer=self.return_layer).squeeze()
                else:
                    img_vec = self.dnn(_image.unsqueeze(0), chan_avg=self.chan_avg
                                       ).squeeze()

            if self.chan_avg:
                # if the vector is channel-wise pooled
                assert len(img_vec) == self.emb, (
                    f'vector length {len(img_vec)}, while expecting {self.emb}.')
            else:
                # if the returned vector is not channel-wise pooled, do spatial pool
                if self.spatial_pool == 'max':
                    img_vec = img_vec.max(0)[0]
                elif self.spatial_pool == 'mean':
                    img_vec = img_vec.mean(0)
                else:
                    max_vec = img_vec.max(0)[0]
                    mean_vec = img_vec.mean(0)
                    img_vec = max_vec / max_vec.max() + 2 * mean_vec / mean_vec.max()
                if verbose:
                    sample['img_vec_2d'] = transforms.Resize(
                        _image.shape[1:])(img_vec[None, ...]).squeeze()
                img_vec = img_vec.flatten()

            img_vec = thresholding(img_vec, self.threshold)
            img_vec = img_vec.float().to(device)

            img_vec = self._scale_(img_vec)
            sample['img_vec'] = img_vec

        ##### fMRI and fmri-mapped emb #####
        if self.load_fmri or self.load_vec_mapped:
            sample['fmri'] = _load_fmri_forward(idx, self.fmri_dir, self.fmri_files,
                self.fmri_pad, self.left_pad, self.right_pad, fmri_model=self.fmri_model,
                fmri_model_args=self.fmri_model_args, extra_fmri_fn=self.extra_fmri_fn,
                verbose=verbose)
            if self.load_vec_mapped:
                with torch.no_grad():
                    sample['img_vec_mapped'] = self.mapper(sample['fmri']).squeeze()
        ##### cat #####
        if self.load_cat:
            cur_cat = self.nsd_cat[str(self.stim_info['cocoId'][nsdId])]
            cat = torch.zeros(self.num_class)
            for c in cur_cat:
                cat[self.cat_list.index(c)] = 1
            if verbose:
                print('Categories:', cat)
            sample['cat'] = cat.to(device)

        return sample


class NSDwithCLIP(Dataset):
    def __init__(self, load_fmri=True, fmri_pad=None, roi=None,
                 fmri_model=None, fmri_model_args={}, extra_fmri_fn=None,
                 load_img=False, img_trans=0.9,
                 load_clip=True, CLIP=None, threshold=0.0, clip_norm=False, clip_std=False, clip_01=False,
                 load_caption=False, caption_selection='avg',
                 load_cat=False, cat_type='things_stuff',
                 load_clip_mapped=False, mapper=None,
                 ):

        """ MAIN DATASET FUNCTION USED """
        '''
        - load_fmri, load_img, load_clip, load_caption, load_cat, load_clip_mapped: 
          all bool, choose the modalities you need. By default load fMRI and CLIP.
          (load_clip means whether loading CLIP image vectors,
           load_caption means whether loading CLIP caption vectors,
           load_clip_mapped means whether loading fMRI-mapped CLIP vectors.)
        - fmri_pad: int, pad fMRI vector to this length. Useful when model
                    requires inputs to have a certain shape.        
        - roi: string, the directory contains extracted ROI voxels.
               if None, return the 3d fmri activity: (83, 104, 81) for 1.8mm.
        - fmri_model: a PyTorch model that takes in fMRI signal as input. fMRI
                      will be passed through this model as the dataset point.
        - fmri_model_args: args for forward loop of fmri_model.
        - extra_fmri_fn: extra processing steps (after passing through fmri_model).
        - img_trans: a value between 0-1. If 0, only apply resizing and to tensor.
                     If > 0, this p controls the probability that an augmentation
                     images will be augmented with probability p before passed to CLIP.
        - CLIP: CLIP model, if None, load the pre computed vectors.
        - threshold: to thresholding CLIP vector values, set to <= 0 if no
                     thresholding is wanted (set to 1.5 to remove spikes)
        - clip_norm: bool. If True, the CLIP vector will be normalized to a
                     unit sphere.
        - clip_std: bool. If True, the CLIP vector will be standardized.
        - clip_01: bool. If True, the CLIP vector will be normalized to 0-1.
        - caption_selection: either 'first' or 'avg' or 'random'. Each image has 
                            multiple captions. 'first' will only keep the first,
                            and 'avg' will average all captions, 'random' will
                            randomly choose one.
        - cat_type: the type of COCO categories, select from "things", "stuff",
                    "things_stuff" (default).
        - mapper: the mapper model to map fMRI to CLIP vectors.
        '''
        assert load_fmri or load_img or load_clip or load_caption or load_cat, (
            'You must choose to load at least one modeality!'
            )

        self.load_fmri = load_fmri
        self.load_img = load_img
        self.load_clip = load_clip
        self.load_caption = load_caption
        self.load_cat = load_cat
        self.load_clip_mapped = load_clip_mapped
        
        stim_order = loadmat(STIM_ORDER_FILE)
        self.subjectim = stim_order['subjectim']
        self.masterordering = stim_order['masterordering']
        self.stim_file = STIM_FILE
        self.img_trans = img_trans
        self.CLIP = CLIP

        if load_fmri or load_clip_mapped:
            self.fmri_dir, self.fmri_files, self.left_pad, self.right_pad, self.fmri_model = (
                _load_fmri_init(roi=roi, fmri_pad=fmri_pad, fmri_model=fmri_model))
            self.fmri_model_args = fmri_model_args
            self.fmri_pad = fmri_pad
            self.extra_fmri_fn = extra_fmri_fn
            if load_clip_mapped:
                assert mapper is not None
                self.mapper = mapper.to(device)
                self.mapper.eval()

        if load_clip:
            if CLIP: # calculate clip vectors on the go
                if load_caption:
                    vetted_cap_path = os.path.join(DATA_DIR, 'nsddata_stimuli/stimuli/nsd/annotations/', f'nsd_captions_vetted_{SUBJ}.pkl')
                    with open(vetted_cap_path, 'rb') as f:
                        self.cap_vetted = pickle.load(f)
            else: # load pre-computed clip vectors
                clip_image_f_path = os.path.join(DATA_DIR, 'clip_features', 'nsd_images.pkl')
                with open(clip_image_f_path, 'rb') as f:
                    self.clip_image_f = pickle.load(f)
                if load_caption:
                    clip_cap_f_path = os.path.join(DATA_DIR, 'clip_features', 'nsd_captions.pkl')
                    with open(clip_cap_f_path, 'rb') as f:
                        self.clip_cap_f = pickle.load(f)

            self.threshold = threshold
            assert (clip_norm and clip_std and clip_01) is False, (
                'normalization or standardization cannot be applied together.')
            self.clip_norm = clip_norm
            self.clip_std = clip_std
            self.clip_01 = clip_01

        if load_caption:
            assert caption_selection in ['first', 'random', 'avg'], (
                "you must choose from 'first', 'random', 'avg' as your caption selection method")
            self.caption_selection = caption_selection

        if load_cat:
            self.cat_list, self.stim_info, self.nsd_cat, self.num_class = _load_cat_init(cat_type)

    def __len__(self):
        try:
            return TRIAL_PER_SESS * len(self.fmri_files)
        except:
            return TRIAL_PER_SESS * 37 # TODO: hard-coded for now, using # sessions for one suject
    
    def _get_caption(self, cap, method):
        if cap.dim() > 1:
            if method == 'first':
                cap = cap[0]
            elif method == 'random':
                rand_id = np.random.choice(len(cap))
                cap = cap[rand_id]
            else:
                cap = cap.mean(0)
        return cap

    def _scale_(self, vec):
        ''' 
        Without thresholding:
        min, max of CLIP vecs are: -9.9688; 5.1289
        mean, std of CLIP vecs are: -0.0045; 0.4602

        With +- 1.5 thresholding:
        mean, std of CLIP vecs are:
        '''
        if self.clip_norm:
            vec = vec / torch.norm(vec, p=2, dim=-1)
        if self.clip_std:
            if self.threshold:
                if self.threshold == 1.5:
                    clip_mean = 0.0047
                    clip_std = 0.3623
                else: # TODO: not the correct way to do, but...
                    clip_mean = vec.mean(-1)
                    clip_std = vec.std(-1)
            else:
                clip_mean = -0.0045
                clip_std = 0.4602
            vec = (vec - clip_mean) / clip_std
        if self.clip_01:
            clip_min = -self.threshold if self.threshold else -9.9688
            clip_max = self.threshold if self.threshold else 5.1289
            vec -= clip_min
            vec /= (clip_max - clip_min)
        return vec

    def get_fmri_shape(self):
        with h5py.File(os.path.join(self.fmri_dir,
                                    self.fmri_files[0]), 'r') as f:
            x = f['betas'][0]
            s = x.shape
            print(f'fmri signal shape: {s}')
            x = torch.FloatTensor(x).to(device)
            if self.fmri_pad:
                x = F.pad(x, (self.left_pad, self.right_pad), 'constant', 0)
                print(f'padded to: {self.fmri_pad}')
            if self.fmri_model is not None:
                with torch.no_grad():
                    x = self.fmri_model(x.unsqueeze(0), **self.fmri_model_args)
            if self.extra_fmri_fn is not None:
                x = self.extra_fmri_fn(x)
            return x.flatten().shape[0]

    def __getitem__(self, idx, verbose=False, load_ori_img=False):
        sample = {}
        nsdId = self.subjectim[int(SUBJ)-1, self.masterordering[idx] - 1] - 1
        sample['nsdId'] = nsdId

        if verbose:
            print(f'stim nsd id: {nsdId}, aka #{nsdId+1} in the tsv files.')

        ##### fMRI #####
        if self.load_fmri or self.load_clip_mapped:
            sample['fmri'] = _load_fmri_forward(idx, self.fmri_dir, self.fmri_files,
                self.fmri_pad, self.left_pad, self.right_pad, fmri_model=self.fmri_model,
                fmri_model_args=self.fmri_model_args, extra_fmri_fn=self.extra_fmri_fn,
                verbose=verbose)

            if self.load_clip_mapped:
                with torch.no_grad():
                    sample['clip_mapped'] = self.mapper(sample['fmri']).squeeze()

        ##### image #####
        if self.load_img:
            with h5py.File(self.stim_file, 'r') as f:
                _image = f['imgBrick'][nsdId]
                sample['img'] = get_img_trans(extra_aug=self.img_trans,
                                              toPIL=False)(_image).to(device)

        ##### clip_image #####
        if self.load_clip:
            if self.CLIP:
                with h5py.File(self.stim_file, 'r') as f:
                    _image = f['imgBrick'][nsdId]
                    # print(f'_image range {_image.min()}, {_image.max()}')
                _image = get_img_trans(extra_aug=self.img_trans)(_image)
                if load_ori_img: sample['_img'] = _image
                _image = self.CLIP[1](_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    clip_image = self.CLIP[0].encode_image(_image).squeeze()
            else:
                clip_image = self.clip_image_f[nsdId]
            clip_image = thresholding(clip_image, self.threshold)
            clip_image = clip_image.float().to(device)

            clip_image = self._scale_(clip_image)
            sample['clip_img'] = clip_image

        ##### clip_caption #####
        if self.load_caption:
            if self.CLIP:
                _cap = clip.tokenize(self.cap_vetted[nsdId]).to(device)
                with torch.no_grad():
                    clip_cap = self._get_caption(self.CLIP[0].encode_text(_cap),
                                                 self.caption_selection
                                                 ).float().to(device)
            else:
                clip_cap = self._get_caption(self.clip_cap_f[nsdId],
                                             self.caption_selection
                                             ).float().to(device)
            clip_cap = thresholding(clip_cap, self.threshold)

            clip_cap = self._scale_(clip_cap)
            sample['clip_cap'] = clip_cap

        ##### categories #####
        if self.load_cat:
            cur_cat = self.nsd_cat[str(self.stim_info['cocoId'][nsdId])]
            cat = torch.zeros(self.num_class)
            for c in cur_cat:
                cat[self.cat_list.index(c)] = 1
            if verbose:
                print('Categories:', cat)
            sample['cat'] = cat.to(device)

        return sample


class NSDwithVQVAE(Dataset):
    def __init__(self, fmri_pad=None, roi=None, img_trans=0.9, vec_norm=False,
        vec_01=False, img_size=256, num_img_tokens=8192):

        '''
        - fmri_pad: int, pad fMRI vector to this length. Useful when model
                    requires inputs to have a certain shape.
        - roi: string, the directory contains extracted ROI voxels.
               if None, return the 3d fmri activity: (83, 104, 81) for 1.8mm
        - img_trans: a value between 0-1. If 0, only apply resizing and to tensor.
                     If > 0, this p controls the probability that an augmentation
                     images will be augmented with probability p before passed to CLIP.
        - vec_norm: bool. If True, the codebook vector will be normalized to a
                     unit sphere.
        - vec_01: bool. If True, the codebook vector will be normalized to 0-1.
        '''

        self.fmri_dir = roi if roi else FMRI_DIR
        self.fmri_files = [f for f in os.listdir(self.fmri_dir) if
                            os.path.isfile(os.path.join(self.fmri_dir, f)) and
                            f[-5:] == '.hdf5']
        self.fmri_pad = fmri_pad

        stim_order = loadmat(STIM_ORDER_FILE)
        self.subjectim = stim_order['subjectim']
        self.masterordering = stim_order['masterordering']
        self.stim_file = STIM_FILE

        self.num_img_tokens = num_img_tokens
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        vqvae = DiscreteVAE(
            image_size = img_size,
            num_layers = 3,           # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
            num_tokens = num_img_tokens,        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
            codebook_dim = 512,       # codebook dimension
            hidden_dim = 64,          # hidden dimension
            num_resnet_blocks = 2,    # number of resnet blocks
            temperature = 0.9,        # gumbel softmax temperature, the lower this is, the harder the discretization
            straight_through = False, # straight-through for gumbel softmax. unclear if it is better one way or the other
            normalization = (img_mean, img_std),
            ).to(device)
        # print(model)
        load_trained_model('/home/sikun/bold5k/data/weights/nsd_vqvae_256img8192.pth', vqvae)
        self.vqvae = vqvae

        self.img_trans = img_trans
        assert (vec_norm and vec_01) is False
        self.vec_norm = vec_norm
        self.vec_01 = vec_01

    def __len__(self):
        return TRIAL_PER_SESS * len(self.fmri_files)

    def _scale_(self, vec):
        '''
        Without thresholding:
        min, max of CLIP vecs are: -9.9688; 5.1289
        mean, std of CLIP vecs are: -0.0045; 0.4602

        With +- 1.5 thresholding:
        mean, std of CLIP vecs are:
        '''
        if self.vec_norm:
            vec = vec / torch.norm(vec, p=2, dim=-1)
        elif self.vec_01:
            # _min = 0
            _max = self.num_img_tokens #- 1
            # vec -= _min
            # vec /= (_max - _min)
            vec /= _max
        return vec

    def get_fmri_shape(self):
        with h5py.File(os.path.join(self.fmri_dir,
                                    self.fmri_files[0]), 'r') as f:
            s = f['betas'][0].shape
            print(f'shape: {s}')
            num_vox = f['betas'][0].flatten().shape[0]
            print(f'num voxel: {num_vox}')
            if self.fmri_pad:
                print(f'padded to: {self.fmri_pad}')
                return self.fmri_pad
            return num_vox

    def __getitem__(self, idx, verbose=False, load_ori_img=False):
        sample = {}
        nsdId = self.subjectim[int(SUBJ)-1, self.masterordering[idx] - 1] - 1
        sample['nsdId'] = nsdId

        if verbose:
            print(f'stim nsd id: {nsdId}, aka #{nsdId+1} in the tsv files.')

        ##### image #####
        with h5py.File(self.stim_file, 'r') as f:
            _image = f['imgBrick'][nsdId]
            # print(f'_image range {_image.min()}, {_image.max()}')
            _image = get_img_trans(extra_aug=self.img_trans, toPIL=False)(_image)
            if load_ori_img: sample['img'] = _image

        with torch.no_grad():
            logits = self.vqvae(_image[None].to(device), return_logits=True)
            img_codebook = logits.argmax(dim=1).flatten(1).squeeze().float()

        # img_codebook = img_codebook.float().to(device)
        img_codebook = self._scale_(img_codebook)
        sample['vqvae'] = img_codebook

        ##### fMRI #####
        sess = idx // TRIAL_PER_SESS + 1
        fmri_file = os.path.join(
            self.fmri_dir, f'{self.fmri_files[0][:-7]}{sess:02}.hdf5')
        with h5py.File(fmri_file, 'r') as f:
            fmri_sample = f['betas'][idx % TRIAL_PER_SESS]
        if verbose:
            print('fmri loaded from', fmri_file)
            print('fmri shape:', fmri_sample.shape)
            print('beta min, mean, max:', fmri_sample.min(),
                fmri_sample.mean(), fmri_sample.max())

        fmri_sample = torch.FloatTensor(fmri_sample).to(device)
        if self.fmri_pad:
            left_pad = (self.fmri_pad - fmri_sample.shape[-1]) // 2
            right_pad = self.fmri_pad - fmri_sample.shape[-1] - left_pad
            fmri_sample = F.pad(fmri_sample, (left_pad, right_pad),
                                'constant', 0)
        sample['fmri'] = fmri_sample

        return sample
