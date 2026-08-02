import json
import math
import os
import pickle
import re

import imageio
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets import register


def _iter_with_progress(root_path, filenames, cache, less_memory=False):
    if cache == 'none':
        return filenames

    root_name = os.path.basename(os.path.normpath(root_path)) or root_path
    desc = f'loading {root_name}'
    if cache == 'bin':
        desc = f'caching {root_name}'
    return tqdm(filenames, desc=desc, leave=False)


def _list_tiff_files(root_path, split_file=None, split_key=None, sort_index=False):
    filenames = [
        file_name
        for file_name in os.listdir(root_path)
        if file_name.lower().endswith((".tif", ".tiff"))
    ]
    if split_file is not None:
        with open(split_file, "r", encoding="utf-8") as file_obj:
            filenames = json.load(file_obj)[split_key]
    elif sort_index:
        filenames = sorted(filenames, key=_extract_numeric_suffix)
    else:
        filenames = sorted(filenames, key=lambda file_name: file_name[:-4])
    return filenames


def _extract_numeric_suffix(filename):
    match = re.search(r"(\d+)(?=\.[^.]+$)", filename)
    if match is None:
        return filename
    return int(match.group(1))


def _load_volume(file_path, less_memory=False):
    if less_memory:
        return torch.tensor(np.array(imageio.imread(file_path), dtype=np.uint16))
    return torch.tensor(np.array(imageio.volread(file_path), dtype=np.float32))


def _load_bin_volume(file_path):
    with open(file_path, "rb") as file_obj:
        cached = pickle.load(file_obj)
    cached = np.ascontiguousarray(cached.transpose(2, 0, 1))
    return torch.from_numpy(cached).float() / 255


def _get_bin_file(root_path, filename):
    bin_root = os.path.join(os.path.dirname(root_path), "_bin_" + os.path.basename(root_path))
    os.makedirs(bin_root, exist_ok=True)
    return os.path.join(bin_root, filename.split(".")[0] + ".pkl")


def _prepare_cached_entry(root_path, filename, cache, less_memory=False):
    file_path = os.path.join(root_path, filename)
    if cache == "none":
        return file_path
    if cache == "bin":
        bin_file = _get_bin_file(root_path, filename)
        if not os.path.exists(bin_file):
            with open(bin_file, "wb") as file_obj:
                pickle.dump(imageio.imread(file_path), file_obj)
        return bin_file
    if cache == "in_memory":
        return _load_volume(file_path, less_memory=less_memory)
    raise ValueError(f"Unsupported cache mode: {cache}")


def _resolve_cached_entry(entry, cache, less_memory=False):
    if cache == "none":
        return _load_volume(entry, less_memory=less_memory)
    if cache == "bin":
        return _load_bin_volume(entry)
    if cache == "in_memory":
        if less_memory:
            return entry.to(torch.float32)
        return entry
    raise ValueError(f"Unsupported cache mode: {cache}")


def _build_sliding_windows(entries, seq_len):
    windows = []
    for begin_idx in range(len(entries) - seq_len + 1):
        windows.append(entries[begin_idx: begin_idx + seq_len])
    return windows


@register('image-folder')
class ImageFolder(Dataset):
    """Load independent 3D TIFF volumes from a single directory."""

    def __init__(self,  root_path_1, tag = None, split_file=None, split_key=None, first_k=None, last_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        filenames = _list_tiff_files(root_path_1, split_file=split_file, split_key=split_key)
        if tag == 'beads':
            filenames = sorted(filenames, key=lambda x: int(x[5:-8]))
        elif tag == 'test_No':
            filenames = sorted(filenames, key=lambda x: int(re.search(r'\d+', x).group()))
        if first_k is not None:
            filenames = filenames[:first_k]
        if last_k is not None:
            filenames = filenames[-last_k:]

        self.files = []
        for filename in _iter_with_progress(root_path_1, filenames, cache):
            entry = (
                _prepare_cached_entry(root_path_1, filename, cache)
                if cache != 'in_memory'
                else torch.tensor(np.array(tifffile.imread(os.path.join(root_path_1, filename)), dtype=np.float32))
            )
            self.files.append(entry)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        return _resolve_cached_entry(self.files[idx % len(self.files)], self.cache)


@register('image-folder-v2')
class ImageFolderV2(Dataset):
    """Load volumes from multiple roots and optionally cache them in memory."""

    def __init__(self, root_path_list, split_file=None, split_key=None, first_k=None, last_k=None,
                 repeat=1, cache='none', sort_index=False, less_memory=False, sample_factor=1):
        self.repeat = repeat
        self.cache = cache
        self.files = []
        self.max_value = 0
        self.less_memory = less_memory
        self.sample_factor = sample_factor
        for root_path in root_path_list:
            filenames = _list_tiff_files(
                root_path,
                split_file=split_file,
                split_key=split_key,
                sort_index=sort_index,
            )
            if first_k is not None:
                filenames = filenames[:first_k]
            if last_k is not None:
                filenames = filenames[-last_k:]
            for filename in _iter_with_progress(root_path, filenames, cache, less_memory=self.less_memory):
                entry = _prepare_cached_entry(root_path, filename, cache, less_memory=self.less_memory)
                if cache == 'in_memory':
                    self.max_value = max(self.max_value, torch.max(entry).item())
                self.files.append(entry)

    def __len__(self):
        return math.ceil(len(self.files) / self.sample_factor) * self.repeat

    def __getitem__(self, idx):
        entry = self.files[idx % len(self.files)]
        return _resolve_cached_entry(entry, self.cache, less_memory=self.less_memory)
        
    def get_max_value(self):
        return self.max_value



@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
        if len(self.dataset_1) != len(self.dataset_2):
            raise ValueError(f'len(self.dataset_1) != len(self.dataset_2)')


    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]

    
    def get_max_value_dataset_1(self):
        return self.dataset_1.get_max_value()
    
    def get_max_value_dataset_2(self):
        return self.dataset_2.get_max_value()



@register('image-folder-seq')
class ImageFolderSeq(Dataset):
    """Historical sequential loader that keeps track of per-root boundaries."""

    def __init__(self, root_path_list, split_file=None, split_key=None, first_k=None, last_k=None,
                 repeat=1, cache='none', sort_index=False):
        self.repeat = repeat
        self.cache = cache
        self.files = []
        self.data_boundary = []
        cnt = 0
        for root_path in root_path_list:
            filenames = _list_tiff_files(
                root_path,
                split_file=split_file,
                split_key=split_key,
                sort_index=sort_index,
            )
            if first_k is not None:
                filenames = filenames[:first_k]
            if last_k is not None:
                filenames = filenames[-last_k:]
            left_cnt = cnt
            for filename in _iter_with_progress(root_path, filenames, cache):
                cnt += 1
                self.files.append(_prepare_cached_entry(root_path, filename, cache))
            right_cnt = cnt
            for filename in filenames:
                self.data_boundary.append([left_cnt, right_cnt])

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        return _resolve_cached_entry(self.files[idx % len(self.files)], self.cache)
        


@register('paired-image-folders-seq')
class PairedImageFoldersSeq(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolderSeq(root_path_1, **kwargs)
        self.dataset_2 = ImageFolderSeq(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
    
    def get_data_boundary(self, idx):
        return self.dataset_1.data_boundary[idx]
    



@register('image-folder-seq-v2')
class ImageFolderSeqV2(Dataset):
    """Build fixed-length temporal windows from consecutive TIFF volumes."""

    def __init__(self, root_path_list, split_file=None, split_key=None, first_k=None, last_k=None,
                 repeat=1, cache='none', sort_index=False, seq_len=4, less_memory=False):
        self.repeat = repeat
        self.cache = cache
        self.files = []
        self.max_value = 0
        self.less_memory = less_memory
        for root_path in root_path_list:
            filenames = _list_tiff_files(
                root_path,
                split_file=split_file,
                split_key=split_key,
                sort_index=sort_index,
            )
            if first_k is not None:
                filenames = filenames[:first_k]
            if last_k is not None:
                filenames = filenames[-last_k:]

            files = []
            for filename in _iter_with_progress(root_path, filenames, cache, less_memory=self.less_memory):
                entry = _prepare_cached_entry(root_path, filename, cache, less_memory=self.less_memory)
                if cache == 'in_memory':
                    self.max_value = max(self.max_value, torch.max(entry).item())
                files.append(entry)

            self.files.extend(_build_sliding_windows(files, seq_len))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        entry = self.files[idx % len(self.files)]
        if self.cache == 'in_memory':
            if self.less_memory:
                return entry.to(torch.float32)
            return entry

        frames = [
            _resolve_cached_entry(frame_entry, self.cache, less_memory=self.less_memory)
            for frame_entry in entry
        ]
        return torch.stack(frames, dim=0)
        
    def get_max_value(self):
        return self.max_value
        


@register('paired-image-folders-seq-v2')
class PairedImageFoldersSeqV2(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolderSeqV2(root_path_1, **kwargs)
        self.dataset_2 = ImageFolderSeqV2(root_path_2, **kwargs)
        if len(self.dataset_1) != len(self.dataset_2):
            raise ValueError(f'len(self.dataset_1) != len(self.dataset_2)')


    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]

    def get_max_value_dataset_1(self):
        return self.dataset_1.get_max_value()
    
    def get_max_value_dataset_2(self):
        return self.dataset_2.get_max_value()



@register('paired-image-folders-seq-v2-random-sample')
class PairedImageFoldersSeqV2RandomSample(Dataset):

    def __init__(self, root_path_1, root_path_2, sample_factor=4, **kwargs):
        self.dataset_1 = ImageFolderSeqV2(root_path_1, **kwargs)
        self.dataset_2 = ImageFolderSeqV2(root_path_2, **kwargs)
        self.sample_factor = sample_factor
        if len(self.dataset_1) != len(self.dataset_2):
            raise ValueError(f'len(self.dataset_1) != len(self.dataset_2)')

    def __len__(self):
        return math.ceil(len(self.dataset_1) / self.sample_factor)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
    
    def get_max_value_dataset_1(self):
        return self.dataset_1.get_max_value()
    
    def get_max_value_dataset_2(self):
        return self.dataset_2.get_max_value()
 
