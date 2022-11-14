
import os
import ipdb
import json
import h5py
import zipfile
import PIL.Image
import numpy as np

import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        use_clip    = False,
        threshold = 0,          # threshold of clip features, set to <= 0 to disable
        normalize_clip = False, # whether to normalize the vector to a unit sphere
        use_fmri    = False,    # Whether to load fmri: for e2e training, load fmri and finetune the mapper together
        fmri_pad    = 0,        # Padding for fmri vector
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        ratio = 1.0,            # how many text-image pairs will be used (0.5 means 0.5 image-text pairs + 0.5 fake pairs. Note if one want to use only 0.5 image-text pairs without using the rest images, set max_size)
        ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._use_clip = use_clip
        self._normalize_clip = normalize_clip
        self._use_fmri = use_fmri
        self._fmri_pad = fmri_pad
        self._raw_labels = None
        self._raw_clip_txt_features = None
        self._raw_clip_img_features = None
        self._label_shape = None
        self._ratio = ratio
        self._threshold = threshold

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def _get_clip_img_features(self, raw_idx):
        if self._raw_clip_img_features is None:
            self._raw_clip_img_features = self._load_clip_img_features(raw_idx) if self._use_clip else None
        return self._raw_clip_img_features

    def _get_clip_txt_features(self, raw_idx):
        if self._raw_clip_txt_features is None:
            self._raw_clip_txt_features = self._load_clip_txt_features(raw_idx) if self._use_clip else None
        return self._raw_clip_txt_features

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_fmri(self, raw_idx):
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def _load_clip_img_features(self):
        raise NotImplementedError

    def _load_clip_txt_features(self):
        raise NotImplementedError

    @staticmethod
    def _thresholding(vec, thr):
        ''' thresholding vec values to [-thr, thr]. '''
        if thr > 0:
            _thr = np.ones_like(vec) * thr
            vec = np.minimum(vec, _thr)
            vec = np.maximum(vec, -_thr)
        return vec

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        if self._use_clip:
            if idx % self._raw_shape[0] > self._ratio*self._raw_shape[0]:
                ipdb.set_trace()
                noise = np.random.normal(0., 1., (512))
                img_fts = self.get_img_features(idx)
                revised_img_fts = 0.25*img_fts/np.linalg.norm(img_fts) + 0.75*noise/np.linalg.norm(noise)
                revised_img_fts = revised_img_fts/np.linalg.norm(revised_img_fts)
                if self._use_fmri:
                    return image.copy(), self.get_label(idx), img_fts, revised_img_fts, self.get_fmri(idx)
                else:
                    return image.copy(), self.get_label(idx), img_fts, revised_img_fts
            else:
                if self._use_fmri:
                    return image.copy(), self.get_label(idx), self.get_img_features(idx), self.get_txt_features(idx), self.get_fmri(idx)
                else:
                    return image.copy(), self.get_label(idx), self.get_img_features(idx), self.get_txt_features(idx)
        else:
            return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_img_features(self, idx):
        img_features = self._get_clip_img_features(self._raw_idx[idx])
        img_features = self._thresholding(img_features, self._threshold)
        if self._normalize_clip:
            img_features = img_features / np.linalg.norm(img_features)
        return img_features.copy()

    def get_txt_features(self, idx):
        try:
            txt_features = self._get_clip_txt_features(self._raw_idx[idx])
            index = np.random.randint(0, len(txt_features), ())
            txt_features = txt_features[index] # randomly select one from the features
            txt_features = np.array(txt_features)
            txt_features = txt_features.astype(np.float32)
            txt_features = self._thresholding(txt_features, self._threshold)
            if self._normalize_clip:
                # TODO: hard coded for now
                if len(txt_features) == 512:
                    txt_features = txt_features / np.linalg.norm(txt_features)
                else:
                    txt_features[:512] = txt_features[:512] / np.linalg.norm(txt_features[:512])
                    txt_features[512:1024] = txt_features[512:1024] / np.linalg.norm(txt_features[512:1024])
                    if len(txt_features) > 1024:
                        txt_features[1024:] = txt_features[1024:] / np.linalg.norm(txt_features[1024:])

            return txt_features.copy()
        except:
            ipdb.set_trace()
            return np.random.normal(0., 1., (512))

    def get_fmri(self, idx):
        fmri = self._load_raw_fmri(self._raw_idx[idx])
        if self._fmri_pad > 0:
            left_pad = (self._fmri_pad - fmri.shape[-1]) // 2
            right_pad = self._fmri_pad - fmri.shape[-1] - left_pad
            fmri = np.pad(fmri, (left_pad, right_pad), 'constant')
        return fmri

    def get_ids(self, idx):
        return (self.nsd_clip['fmriId'][self._raw_idx[idx]],
                self.nsd_clip['nsdId'][self._raw_idx[idx]],
                self.nsd_clip['cocoId'][self._raw_idx[idx]])

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution = None,      # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
            self.json_name = 'dataset.json'

        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
            self.json_name = 'dataset.json'
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = self.json_name
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def _load_clip_img_features(self, raw_idx):
        fname = self.json_name
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            clip_features = json.load(f)['clip_img_features']
        if clip_features is None:
            return None
        clip_features = dict(clip_features)
        clip_features = [clip_features[fname.replace('\\', '/')] for fname in self._image_fnames]
        clip_features = np.array(clip_features)
        clip_features = clip_features.astype(np.float64)
        return clip_features[raw_idx]

    def _load_clip_txt_features(self, raw_idx):
        fname = self.json_name
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            clip_features = json.load(f)['clip_txt_features']
        if clip_features is None:
            return None
        clip_features = dict(clip_features)
        clip_features = [clip_features[fname.replace('\\', '/')] for fname in self._image_fnames]
        return clip_features[raw_idx]
#----------------------------------------------------------------------------

class NsdClipDataset(Dataset):
    def __init__(self,
        path,                   # json file contains mappings to fmriId, nsdId, cocoId
        data_dir = None,        # root data folder
        fmri_dir = None,
        img_file = None,
        clip_img_file = None,   # pre-computed clip image vectors
        clip_cap_file = None,   # pre-computed clip caption vectors, OR pre-computed conditions
        use_mapped  = False,    # Whether to use precomputed fmri-mapped vector, choose from None (will use gt txt clip vec), 'img', 'cap', 'add', 'cat', 'mix' (the latter three combines clip_img and clip_cap), 'all' (will concat img, cap, cap_gt)
        resolution = None,      # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
        ):

        self._path = path
        with open(path, 'r') as f:
            self.nsd_clip = json.load(f)

        data_dir = self.default(data_dir, '/home/sikun/NSD/data/')
        fmri_dir = self.default(fmri_dir, os.path.join(data_dir,
            'nsddata_betas/ppdata/subj01/func1pt8mm/nsdgeneral_zscored'))
        self.fmri_dir = fmri_dir
        self.fmri_files = [f for f in os.listdir(fmri_dir) if
            os.path.isfile(os.path.join(fmri_dir, f)) and self._file_ext(f) == '.hdf5']

        self.img_file = self.default(img_file, os.path.join(data_dir,
            'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5'))

        self.clip_img_file = self.default(clip_img_file,
            os.path.join(data_dir, 'clip_features', 'nsd_images.json'))    
        self.clip_cap_file = self.default(clip_cap_file,
            os.path.join(data_dir, 'clip_features', 'nsd_captions.json'))

        self.mapped_clip_img_file = self.default(clip_cap_file,
            os.path.join(data_dir, 'clip_features', 'fmri_nsd_images.json'))
        self.mapped_clip_cap_file = self.default(clip_cap_file,
            os.path.join(data_dir, 'clip_features', 'fmri_nsd_captions.json'))
        allowed_ks = ['img', 'cap', 'add', 'cat', 'mix', 'all']
        assert (use_mapped is None) or (use_mapped in allowed_ks), (
            f'Use_mapped must be None, or one of {allowed_ks}')
        self._use_mapped = use_mapped

        name = os.path.splitext(os.path.basename(path))[0]
        raw_shape = [len(self.nsd_clip['fmriId'])] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def exists(val):
        return val is not None

    def default(self, val, d):
        return val if self.exists(val) else d

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_raw_fmri(self, raw_idx):
        trial_per_sess=750
        _fmriId = self.nsd_clip['fmriId'][raw_idx]
        sess = _fmriId // trial_per_sess + 1
        fmri_file = os.path.join(self.fmri_dir,
            f'{self.fmri_files[0][:-7]}{sess:02}.hdf5')
        with h5py.File(fmri_file, 'r') as f:
            _fmri = f['betas'][_fmriId % trial_per_sess]
        return _fmri

    def _load_raw_image(self, raw_idx):
        _nsdId = self.nsd_clip['nsdId'][raw_idx]
        with h5py.File(self.img_file, 'r') as f:
            image = f['imgBrick'][_nsdId]
        # resize to 256 (hardcoded for now... can adapt to resolution if needed)
        image = PIL.Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _get_clip_id(self, raw_idx, use_nsdId=True):
        # return the key of clip features in the precomputed json files
        return str(self.nsd_clip['nsdId'][raw_idx] if use_nsdId else (
            self.nsd_clip['fmriId'][raw_idx]))

    def _load_clip_img_features(self, raw_idx):
        with open(self.clip_img_file, 'r') as f:
            clip_features = json.load(f)[self._get_clip_id(raw_idx)]

        clip_features = np.array(clip_features)
        clip_features = clip_features.astype(np.float64)

        return clip_features

    def _load_clip_txt_features(self, raw_idx):
        if self._use_mapped is None:
            with open(self.clip_cap_file, 'r') as f:
                c = json.load(f)[self._get_clip_id(raw_idx)]
        elif self._use_mapped == 'img':
            with open(self.mapped_clip_img_file, 'r') as f:
                clip_features = json.load(f)[self._get_clip_id(raw_idx, use_nsdId=False)]
            clip_features = [clip_features[1]] # corresponding json saves (nsdId, clipfeatures)
        elif self._use_mapped == 'cap':
            with open(self.mapped_clip_cap_file, 'r') as f:
                clip_features = json.load(f)[self._get_clip_id(raw_idx, use_nsdId=False)]
            clip_features = [clip_features[1]] # corresponding json saves (nsdId, clipfeatures)
        else:
            with open(self.mapped_clip_img_file, 'r') as f:
                mapped_clip_img = json.load(f)[self._get_clip_id(raw_idx, use_nsdId=False)][1]
            with open(self.mapped_clip_cap_file, 'r') as f:
                mapped_clip_cap = json.load(f)[self._get_clip_id(raw_idx, use_nsdId=False)][1]
            if self._use_mapped == 'add':
                assert len(mapped_clip_img) == len(mapped_clip_cap)
                clip_features = [0.5 * mapped_clip_img[i] + 0.5 * mapped_clip_cap[i]
                                 for i in range(len(mapped_clip_img))]
            elif self._use_mapped == 'cat':
                clip_features = mapped_clip_img + mapped_clip_cap
            elif self._use_mapped == 'mix':
                clip_features = mapped_clip_cap.copy()
                # randomly select half of the positions to use img vec
                img_pos = np.random.choice(range(len(clip_features)), size=len(clip_features)//2, replace=False) # TODO: mix here or have a fixed idx in init?
                for _img_pos in img_pos:
                    clip_features[_img_pos] = mapped_clip_img[_img_pos]
            elif self._use_mapped == 'all':
                with open(self.clip_cap_file, 'r') as f:
                    clip_features = json.load(f)[self._get_clip_id(raw_idx)]
                    clip_features = clip_features[np.random.randint(0, len(clip_features), ())]
                clip_features = mapped_clip_img + mapped_clip_cap + clip_features
            clip_features = [clip_features]

        return clip_features

    def _load_raw_labels(self): # not using
        return None
