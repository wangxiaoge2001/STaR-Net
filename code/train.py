import argparse
import copy
import os
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import math
import random
import datasets
import models
import utils
import numpy as np

L1_pixelwise = torch.nn.L1Loss()
L2_pixelwise = torch.nn.MSELoss()





def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_tv_depth_loss(volume):

    gv = list(torch.gradient(volume.squeeze(), dim=[-3]) )

    loss = sum([( gv[ii].abs()  ).mean() for ii in range(len(gv)) ])
    return loss





def loss_ssim_vol(img1, img2):
    ret = torch.zeros((1, ), device=img1.device)
    for i in range(img1.shape[1]):
        ret = ret + loss_fn_ssim((img1[:, i, :, :]).unsqueeze(0), (img2[:, i, :, :]).unsqueeze(0))
    return ret / img1.shape[1]




def loss_lpips_xy(img1, img2):
    ret = torch.zeros((1, ), device=img1.device)
    for i in range(img1.shape[1]):
        ret = ret + loss_fn_vgg(img1[:, i, :, :], img2[:, i, :, :])
    return ret / img1.shape[1]


def loss_lpips_xy_weight(img1, img2, z_axial_weight):
    ret = torch.zeros((1, ), device=img1.device)
    for i in range(img1.shape[1]):
        ret = ret + z_axial_weight[i] * loss_fn_vgg(img1[:, i, :, :], img2[:, i, :, :])
    return ret / sum(z_axial_weight)



def loss_lpips_axial(img1, img2):
    ret = torch.zeros((1, ), device=img1.device)
    for i in range(img1.shape[2]):
        ret = ret + loss_fn_vgg(img1[:, :, i, :], img2[:, :, i, :])
    ret = ret / img1.shape[2]
    for i in range(img1.shape[3]):
        ret = ret + loss_fn_vgg(img1[:, :, :, i], img2[:, :, :, i])
    ret = ret / img1.shape[3]
    return ret


def loss_lpips_3d(img1, img2):
    return (loss_lpips_xy(img1, img2) + loss_lpips_axial(img1, img2)) / 2


def loss_lpips_(img1, img2):
    ret = torch.zeros((1, ), device=img1.device)
    for i in range(img1.shape[2]):
        ret = ret + loss_fn_vgg(img1[:, :, i, :], img2[:, :, i, :])
    return ret / img1.shape[2]


def loss_lpips_mip(img1, img2):
    ret = torch.zeros((1, ), device=img1.device)
    for dim in range(1, 4):
        img1_mip, img1_idx = torch.max(img1, dim)
        img2_mip, img2_idx = torch.max(img2, dim)
        ret = ret + loss_fn_vgg(img1_mip, img2_mip)
    return ret / 3

def loss_lpips_mip_xy(img1, img2):
    img1_mip, img1_idx = torch.max(img1, 1)
    img2_mip, img2_idx = torch.max(img2, 1)
    ret = loss_fn_vgg(img1_mip, img2_mip)
    return ret

def normalize_slice(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))

def normalize_slice_v2(img):
    return img / torch.max(img)


def loss_perceptual_2_5_d(img1, img2, dim, is_norm, is_interpolate):

    ret = torch.zeros((1, ), device=img1.device)
    for i in range(img1.shape[dim]):
        if dim == 1:
            slice1 = img1[:, i, :, :]
            slice2 = img2[:, i, :, :]
        elif dim == 2:
            slice1 = img1[:, :, i, :]
            slice2 = img2[:, :, i, :]
        elif dim == 3:
            slice1 = img1[:, :, :, i]
            slice2 = img2[:, :, :, i]
        else:
            raise Exception(f'illegal dim = {dim} in loss_perceptual_2_5_d')
        if is_interpolate:
            if dim == 2 or dim == 3:
                slice1 = slice1.unsqueeze(0)
                slice1 = nn.functional.interpolate(slice1, size = (slice1.shape[2] * 2,slice1.shape[3]) , mode='bilinear',align_corners=False)
                slice1 = slice1.squeeze(0)
                slice2 = slice2.unsqueeze(0)
                slice2 = nn.functional.interpolate(slice2, size = (slice2.shape[2] * 2,slice2.shape[3]) , mode='bilinear',align_corners=False)
                slice2 = slice2.squeeze(0)









        ret = ret + loss_fn_vgg(slice1, slice2, normalize=is_norm)
    return ret / img1.shape[dim]


def loss_perceptual_v2(img1, img2, is_norm):
    ret = torch.zeros((1, ), device=img1.device)
    z_depth = img1.shape[1]
    z_c = z_depth//3
    for i in range(z_c):
        slice1 = img1[:, 3*i:3*(i + 1), :, :]
        slice2 = img2[:, 3*i:3*(i + 1), :, :]
        ret = ret + loss_fn_vgg(slice1, slice2, normalize=is_norm)
        temp = loss_fn_vgg(slice1, slice2, normalize=is_norm)
        if i == 0:
            loss_lp = temp
        else:
            loss_lp = temp + loss_lp
    loss_sum = loss_lp / z_c

    if z_depth%3 != 0:
        slice1 = img1[:, z_depth-3 :z_depth, :, :]
        slice2 = img1[:, z_depth-3 :z_depth, :, :]
        temp_add = loss_fn_vgg(slice1, slice2, normalize=is_norm)
        loss_sum = (temp_add + loss_lp)/(z_c+1)

    return loss_sum


def loss_perceptual_v3(img1, img2, is_norm, is_mean, extend_batch):
    if is_mean:
        img1 = img1 / (torch.mean(img1) + 1e-5)
        img2 = img2 / (torch.mean(img1) + 1e-5)
    img1 = img1.reshape(extend_batch, -1, *img1.shape[2:])
    img2 = img2.reshape(extend_batch, -1, *img2.shape[2:])
    z_depth = img1.shape[1]
    z_c = z_depth//3
    for i in range(z_c):
        slice1 = img1[:, 3*i:3*(i + 1), :, :]
        slice2 = img2[:, 3*i:3*(i + 1), :, :]
        temp = loss_fn_vgg(slice1, slice2, normalize=is_norm)
        temp = torch.mean(temp, dim=0, keepdim=False)
        if i == 0:
            loss_lp = temp
        else:
            loss_lp = temp + loss_lp
    loss_sum = loss_lp / z_c

    if z_depth%3 != 0:
        slice1 = img1[:, z_depth-3 :z_depth, :, :]
        slice2 = img1[:, z_depth-3 :z_depth, :, :]
        temp_add = loss_fn_vgg(slice1, slice2, normalize=is_norm)
        temp_add = torch.mean(temp, dim=0, keepdim=False)
        loss_sum = (temp_add + loss_lp)/(z_c+1)

    return loss_sum


def loss_perceptual_v4(img1, img2, dim, is_norm, extend_batch):

    extend_batch = 16
    img1 = img1.reshape(extend_batch, -1, *img1.shape[2:])
    img2 = img2.reshape(extend_batch, -1, *img2.shape[2:])
    z_depth = img1.shape[1]
    for i in range(z_depth):
        slice1 = img1[:, [i], :, :]
        slice2 = img2[:, [i], :, :]
        temp = loss_fn_vgg(slice1, slice2, normalize=is_norm)
        temp = torch.mean(temp, dim=0, keepdim=False)
        if i == 0:
            loss_lp = temp
        else:
            loss_lp = temp + loss_lp
    loss_sum = loss_lp / z_depth

    return loss_sum


def loss_perceptual_v4_2_5_d(img1, img2, dim, is_norm, lpips_extend_batch_z, lpips_extend_batch_xy):
    ret = torch.zeros((1, ), device=img1.device)
    if dim == 1:
        extend_batch = lpips_extend_batch_z
    elif dim == 2:
        extend_batch = lpips_extend_batch_xy
        img1 = torch.permute(img1, (0, 2, 1, 3))
        img2 = torch.permute(img2, (0, 2, 1, 3))
    elif dim == 3:
        extend_batch = lpips_extend_batch_xy
        img1 = torch.permute(img1, (0, 3, 1, 2))
        img2 = torch.permute(img2, (0, 3, 1, 2))
    else:
        raise Exception(f'illegal dim = {dim} in loss_perceptual_2_5_d')

    img1 = img1.reshape(extend_batch, -1, *img1.shape[2:])
    img2 = img2.reshape(extend_batch, -1, *img2.shape[2:])

    for i in range(img1.shape[1]):
        slice1 = img1[:, [i], :, :]
        slice2 = img2[:, [i], :, :]
        temp = loss_fn_vgg(slice1, slice2, normalize=is_norm)
        temp = torch.mean(temp, dim=0, keepdim=False)
        ret = ret + temp
    return ret / img1.shape[1]



def loss_perceptual_v4_2_5_d_window(img1, img2, dim, is_norm, lpips_extend_batch_z, lpips_extend_batch_xy, window_size_z, window_size_xy):

    ret = torch.zeros((1, ), device=img1.device)
    if dim == 1:
        extend_batch = lpips_extend_batch_z
        window_size = window_size_z
    elif dim == 2:
        extend_batch = lpips_extend_batch_xy
        img1 = torch.permute(img1, (0, 2, 1, 3))
        img2 = torch.permute(img2, (0, 2, 1, 3))
        window_size = window_size_xy
    elif dim == 3:
        extend_batch = lpips_extend_batch_xy
        img1 = torch.permute(img1, (0, 3, 1, 2))
        img2 = torch.permute(img2, (0, 3, 1, 2))
        window_size = window_size_xy
    else:
        raise Exception(f'illegal dim = {dim} in loss_perceptual_2_5_d')

    if np.random.random() < 0.5:
        img1 = img1[:, window_size//2:-window_size//2, :, :]
        img2 = img2[:, window_size//2:-window_size//2, :, :]

    batch_size = img1.shape[0]

    img1 = img1.reshape(-1, window_size, *img1.shape[2:])
    img2 = img2.reshape(-1, window_size, *img2.shape[2:])
    img1 = torch.sum(img1, dim=1, keepdim=True)
    img2 = torch.sum(img2, dim=1, keepdim=True)
    img1 = img1.reshape(batch_size, -1, *img1.shape[2:])
    img2 = img2.reshape(batch_size, -1, *img2.shape[2:])

    c = img1.shape[1]
    remainder = c % extend_batch
    if remainder != 0:
        pad = extend_batch - remainder
        pad_shape = list(img1.shape)
        pad_shape[1] = pad
        img1 = torch.cat([img1, torch.zeros(pad_shape, dtype=img1.dtype, device=img1.device)], dim=1)
        img2 = torch.cat([img2, torch.zeros(pad_shape, dtype=img2.dtype, device=img2.device)], dim=1)


    img1 = img1.reshape(extend_batch, -1, *img1.shape[2:])
    img2 = img2.reshape(extend_batch, -1, *img2.shape[2:])

    for i in range(img1.shape[1]):
        slice1 = img1[:, [i], :, :]
        slice2 = img2[:, [i], :, :]
        temp = loss_fn_vgg(slice1, slice2, normalize=is_norm)
        temp = torch.mean(temp, dim=0, keepdim=False)
        ret = ret + temp
    return ret / img1.shape[1]

def generate_loss_weight(indices, values):
    array = [values[0]] * indices[-1]
    for i in range(1, len(indices)):
        start = indices[i-1]
        end = indices[i]
        value_start = values[i-1]
        value_end = values[i]
        for j in range(start, end):
            array[j] = value_start + (value_end - value_start) * (j - start) / (end - start)
    return array

def generate_intervals(max_epoch, initial_interval, zero_len, value_list, interval_gamma):
    output = np.zeros(max_epoch, dtype=int)
    interval = initial_interval
    idx = zero_len

    while idx < max_epoch:
        for value in value_list:
            end_idx = min(idx + interval, max_epoch)
            output[idx:end_idx] = value
            idx += interval
            if idx >= max_epoch:
                break
        interval = int(interval / interval_gamma)
        if interval < 1:
            interval = 1
    return output

def build_save_name(args):
    config_stem = Path(args.config).stem
    if args.name is not None:
      run_name = f'{args.name}_{config_stem}'
    else:
      run_name = f'{config_stem}'
    if args.tag is not None:
        run_name += f'_{args.tag}'
    return run_name


def snapshot_code_directory(save_path):
    source_dir = Path(__file__).resolve().parent
    snapshot_dir = Path(save_path) / 'code'
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    shutil.copytree(source_dir, snapshot_dir)


def build_checkpoint(config, model, optimizer, epoch):
    model_spec = copy.deepcopy(config['model'])
    model_spec['sd'] = model.state_dict()
    optimizer_spec = copy.deepcopy(config['optimizer'])
    optimizer_spec['sd'] = optimizer.state_dict()
    return {
        'model': model_spec,
        'optimizer': optimizer_spec,
        'epoch': epoch,
    }


def initialize_runtime_from_config(config):
    if config.get('seed') is not None:
        seed = config.get('seed')
        print(f'set pytorch seed = {seed}')
        seed_everything(seed)

    if config.get('loss_seq_select_num') is not None:
        global seq_selection_weight
        seq_selection_weight = torch.ones(1, config.get('seq_len')) * 1.0

    if config.get('loss_lpips') is not None:
        global loss_fn_vgg
        loss_perceptual_is_single_channel = config.get('loss_perceptual_is_single_channel', False)
        loss_perceptual_is_zero_one_scaling = config.get('loss_perceptual_is_zero_one_scaling', False)
        loss_fn_vgg = utils.LPIPS_vgg16(
            is_single_channel=loss_perceptual_is_single_channel,
            is_zero_one=loss_perceptual_is_zero_one_scaling,
        ).cuda()

    if config.get('loss_lpips_indices') is not None:
        global lpips_loss_weight_array
        lpips_loss_weight_array = generate_loss_weight(
            config.get('loss_lpips_indices'),
            config.get('loss_lpips_scales'),
        )

    if config.get('loss_rTG_indices') is not None:
        global rTG_loss_weight_array
        rTG_loss_weight_array = generate_loss_weight(
            config.get('loss_rTG_indices'),
            config.get('loss_rTG_scales'),
        )

    if config.get('loss_perceptual_2_5_d') is not None:
        global loss_perceptual_dim_list
        if config.get('loss_lpips_indices') is not None:
            tmp_list = config.get('loss_lpips_indices')
            zero_len = tmp_list[1]
        else:
            zero_len = 0

        initial_interval = config.get('initial_interval')
        interval_gamma = config.get('interval_gamma', 1.5)

        if config.get('loss_perceptual_window') is not None:
            global loss_perceptual_window_list
            window_size_value = config.get('window_size_value')
            loss_perceptual_window_list = generate_intervals(
                config.get('epoch_max'),
                int(initial_interval / 2),
                zero_len,
                window_size_value,
                2,
            )
            print(f'window_size_value = {window_size_value}')

        print(f'loss_perceptual_2_5_d: zero_len = {zero_len}, initial_interval = {initial_interval}')
        loss_perceptual_dim_list = generate_intervals(
            config.get('epoch_max'),
            initial_interval,
            zero_len,
            [1, 2, 3],
            interval_gamma,
        )

    if config.get('online_RPN_flag') is not None:
        global online_RPN_gamma, RPN_noise_min_array, RPN_noise_max_array
        RPN_noise_min_array = generate_loss_weight(
            config.get('RPN_noise_min_array_indices'),
            config.get('RPN_noise_min_array_scales'),
        )
        RPN_noise_max_array = generate_loss_weight(
            config.get('RPN_noise_max_array_indices'),
            config.get('RPN_noise_max_array_scales'),
        )
        online_RPN_gamma = config.get('online_RPN_gamma')

    if config.get('online_occlusion_flag') is not None:
        global online_occlusion_gamma, occlusion_min_array, occlusion_max_array
        occlusion_min_array = generate_loss_weight(
            config.get('occlusion_min_array_indices'),
            config.get('occlusion_min_array_scales'),
        )
        occlusion_max_array = generate_loss_weight(
            config.get('occlusion_max_array_indices'),
            config.get('occlusion_max_array_scales'),
        )
        online_occlusion_gamma = config.get('online_occlusion_gamma')




def weighted_random(min_value, max_value, weight, gamma, reverse):



    adjusted_weight = weight ** gamma

    adjusted_rand = random.uniform(0, 1) ** (1 - adjusted_weight)

    if reverse:
        random_value = round(max_value - (max_value - min_value) * adjusted_rand)
    else:
        random_value = round(min_value + (max_value - min_value) * adjusted_rand)

    return random_value


def add_RPN_noise_online(lfstack, epoch, epoch_max, gamma, RPN_noise_min, RPN_noise_max):
    poisson_lambda = weighted_random(RPN_noise_min, RPN_noise_max, (epoch_max + 1 - epoch) / epoch_max, gamma, 0)

    lfstack = lfstack.clamp(0,1)
    lfstack = lfstack * poisson_lambda
    lfstack = torch.poisson(lfstack)
    lfstack = lfstack / poisson_lambda
    lfstack = lfstack.clamp(0,1)
    return lfstack


def generate_occlusion_matrix(n, is_gpu=1):
    matrix_row = np.ones(3*n)
    matrix_col = np.ones(3*n)

    matrix = np.ones((3*n, 3*n))

    small_rect_min_row = np.random.randint(5, 2*n)
    small_rect_max_row = np.random.randint(small_rect_min_row + 1, 3*n - 5)
    small_rect_min_col = np.random.randint(5, 2*n)
    small_rect_max_col = np.random.randint(small_rect_min_col + 1, 3*n - 5)

    large_rect_min_row = np.random.randint(0, small_rect_min_row - 2)
    large_rect_max_row = np.random.randint(small_rect_max_row + 2, 3*n)
    large_rect_min_col = np.random.randint(0, small_rect_min_col - 2)
    large_rect_max_col = np.random.randint(small_rect_max_col + 2, 3*n)

    matrix_row[small_rect_min_row:small_rect_max_row] = 0
    matrix_row[small_rect_max_row:large_rect_max_row] = np.linspace(0, 1, large_rect_max_row - small_rect_max_row)
    matrix_row[large_rect_min_row:small_rect_min_row] = np.linspace(1, 0, small_rect_min_row - large_rect_min_row)

    matrix_col[small_rect_min_col:small_rect_max_col] = 0
    matrix_col[small_rect_max_col:large_rect_max_col] = np.linspace(0, 1, large_rect_max_col - small_rect_max_col)
    matrix_col[large_rect_min_col:small_rect_min_col] = np.linspace(1, 0, small_rect_min_col - large_rect_min_col)

    matrix = np.maximum.outer(matrix_row, matrix_col)
    matrix = matrix[n:2*n, n:2*n]
    if is_gpu:
        matrix = torch.from_numpy(matrix)
        return matrix.cuda()
    else:
        return matrix




def eval_psnr_seq4(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, writer = None, config=None, EPOCH=0, tensorboard_image_writing = True, seq_len=4):
    model.eval()


    if eval_type is None:
        metric_fn = utils.calc_psnr
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    cnt = 0
    for batch in pbar:
        cnt = cnt + 1
        for k, v in batch.items():
            batch[k] = v.cuda()

        if eval_bsize is None:
            with torch.no_grad():
                inp = batch['inp']
                if seq_len == 1:
                    inp = inp.squeeze(1)
                if config.get('input_scale_factor') is not None:
                    inp = inp * config.get('input_scale_factor')
                pred = model(inp, batch['scale'])


        pred = pred.clamp_(0, 1)
        pred[torch.isnan(pred)] = 0
        pred[torch.isinf(pred)] = 0

        gt = batch['gt']
        if config.get('gt_scale_factor') is not None:
            gt = gt * config.get('gt_scale_factor')






        if writer is not None and tensorboard_image_writing:
            seq_num = gt.shape[1]
            inp_list = [batch['inp'][0,seq_id,0,:,:] for seq_id in range(0, seq_num)]
            writer.add_image(f"val_input_{cnt}", utils.cmap[(torch.cat(inp_list,dim=1)*255).long().cpu()] ,dataformats='HWC',global_step=EPOCH)


            pred_list = [pred[0,seq_id,:,:,:].max(0).values for seq_id in range(0, seq_num)]
            writer.add_image(f'val_pred_xy_{cnt}', utils.cmap[(torch.cat(pred_list,dim=1)*255).long().cpu()] ,dataformats='HWC',global_step=EPOCH)


            pred_list = [pred[0,seq_id,:,:,:].max(2).values.permute(1,0) for seq_id in range(0, seq_num)]
            writer.add_image(f'val_pred_yz_{cnt}', utils.cmap[(torch.cat(pred_list,dim=1)*255).long().cpu()] ,dataformats='HWC',global_step=EPOCH)


            gt_list = [batch['gt'][0,seq_id,:,:,:].max(0).values for seq_id in range(0, seq_num)]
            writer.add_image(f'val_gt_xy_{cnt}', utils.cmap[(torch.cat(gt_list,dim=1)*255).long().cpu()] ,dataformats='HWC',global_step=EPOCH)


            gt_list = [batch['gt'][0,seq_id,:,:,:].max(2).values.permute(1,0) for seq_id in range(0, seq_num)]
            writer.add_image(f'val_gt_yz_{cnt}', utils.cmap[(torch.cat(gt_list,dim=1)*255).long().cpu()] ,dataformats='HWC',global_step=EPOCH)

        res = metric_fn(pred, gt)
        val_res.add(res.item(), batch['inp'].shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()



def add_mask_online(lfstack, epoch, epoch_max, gamma, occlusion_min, occlusion_max, angle_num):

    occlusion_num = weighted_random(occlusion_min, occlusion_max, (epoch_max + 1 - epoch) / epoch_max, gamma, 1)
    obscured_views = set()
    for _ in range(occlusion_num):
        obscured_view = np.random.randint(0, angle_num)
        while obscured_view in obscured_views:
            obscured_view = np.random.randint(0, angle_num)
        obscured_views.add(obscured_view)

        occlusion_matrix = generate_occlusion_matrix(lfstack.shape[-1])

        for batch_idx in range(lfstack.shape[0]):
            for seq_id in range(lfstack.shape[1]):
                lfstack[batch_idx, seq_id, obscured_view, :, :] = lfstack[batch_idx, seq_id, obscured_view, :, :] * occlusion_matrix
    return lfstack

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    log('{} dataset: input_max_value={}'.format(tag, dataset.get_max_value_dataset_1()))
    log('{} dataset: gt_max_value={}'.format(tag, dataset.get_max_value_dataset_2()))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = MultiEpochsDataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=4+5*int(tag=='train'), pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def check_model_params_update(model):
    for name, weight in model.named_parameters():

        if weight.requires_grad:



            pass


def prepare_training():

    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()

        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            print(config['multi_step_lr'])
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
            for _ in range(epoch_start - 1):
                lr_scheduler.step()

    elif config.get('load_pretrain') is not None:
        sv_file = torch.load(config['load_pretrain'])
        model = models.make(config['model']).cuda()
        model_dict=model.state_dict()
        pretrained_dict = {k: v for k, v in sv_file['model']['sd'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    log('model details: ')
    log(model)
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer, epoch):

    model.train()

    if config.get('loss_l2') is not None:
        l2_loss_ave = utils.Averager()
    if config.get('loss_l1') is not None:
        l1_loss_ave = utils.Averager()
    if config.get('loss_gradient') is not None:
        gradient_loss_ave = utils.Averager()
    if config.get('loss_ssim') is not None:
        ssim_loss_ave = utils.Averager()
    if config.get('loss_lpips') is not None:
        lpips_loss_ave = utils.Averager()
        global lpips_loss_weight_array
    if config.get('loss_tv') is not None:
        tv_loss_ave = utils.Averager()
    if config.get('loss_rTG') is not None:
        rTG_loss_ave = utils.Averager()
    if config.get('loss_l2_segment') is not None:
        l2_segment_loss_ave = utils.Averager()
    if config.get('loss_optical_flow_tg') is not None:
        optical_flow_tg_loss_ave = utils.Averager()
    if config.get('loss_rTG_lpips') is not None:
        rTG_lpips_loss_ave = utils.Averager()
    if config.get('loss_tv_depth') is not None:
        tv_depth_loss_ave = utils.Averager()

    if config.get('record_each_seq') is not None:
        record_each_seq = []
        for seq_id in range(config.get('seq_len')):
            record_each_seq.append(utils.Averager())


    train_loss = utils.Averager()


    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        if config.get('rand_factor_lowerbound') is not None:
            rand_factor = np.random.uniform(config.get('rand_factor_lowerbound'), 1.0)
        else:
            rand_factor = 1.
        inp = batch['inp']

        if config.get('input_scale_factor') is not None:
            inp = inp * config.get('input_scale_factor')




        if config.get('online_occlusion_flag') is not None:

            inp = add_mask_online(inp, epoch, epoch_max, online_occlusion_gamma, occlusion_min_array[epoch - 1], occlusion_max_array[epoch - 1], config.get('angle_num'))
        if config.get('online_RPN_flag') is not None:

            inp = add_RPN_noise_online(inp, epoch, epoch_max, online_RPN_gamma, RPN_noise_min_array[epoch - 1], RPN_noise_max_array[epoch - 1])

        if config.get('input_seq4') is None:
            inp = inp.reshape(1, -1, *inp.shape[3:])


        gt = batch['gt']*rand_factor
        if config.get('gt_scale_factor') is not None:
            gt = gt * config.get('gt_scale_factor')
        if config.get('seq_len') == 1:
            inp = inp.squeeze(1)



        pred = model(inp*rand_factor, batch['scale'])


        pred[torch.isnan(pred)] = 0
        pred[torch.isinf(pred)] = 0





        loss = 0

        if config.get('loss_rTG') is not None:
            if epoch > config.get('loss_rTG_epoch'):
                rTG_loss_weight = config.get('loss_rTG')
                if config.get('loss_rTG_indices') is not None:
                    global rTG_loss_weight_array
                    rTG_loss_weight = rTG_loss_weight * rTG_loss_weight_array[epoch - 1]

                rTG_mask = torch.abs(utils.get_rTG_3D_v2(torch.squeeze(gt), config.get('seq_len')))
                pred_rTG = torch.squeeze(pred) * rTG_mask
                gt_rTG = torch.squeeze(gt) * rTG_mask

                rTG_loss = rTG_loss_weight * L2_pixelwise(pred_rTG, gt_rTG)
                rTG_loss_ave.add(rTG_loss.item())
                loss = loss + rTG_loss

                if config.get('loss_rTG_lpips') is not None:
                    lpips_loss_weight = config.get('loss_lpips')
                    if config.get('loss_lpips_indices') is not None:

                        lpips_loss_weight = lpips_loss_weight * lpips_loss_weight_array[epoch - 1]

                    rTG_lpips_loss_weight = lpips_loss_weight * rTG_loss_weight
                    if rTG_lpips_loss_weight == 0:
                        rTG_lpips_loss = 0
                        rTG_lpips_loss_ave.add(0)
                    else:
                        if config.get('loss_perceptual_v3') is not None:
                            loss_perceptual_v3_is_mean = False
                            if config.get('loss_perceptual_v3_is_mean') is not None:
                                loss_perceptual_v3_is_mean = config.get('loss_perceptual_v3_is_mean')
                            rTG_lpips_loss = rTG_lpips_loss_weight * loss_perceptual_v3(gt_rTG, pred_rTG, is_norm=False, is_mean=False, extend_batch=config.get('lpips_extend_batch'))
                        elif config.get('loss_perceptual_v4_2_5_d_window') is not None:
                            dim_this = loss_perceptual_dim_list[epoch - 1]
                            rTG_lpips_loss = rTG_lpips_loss_weight * loss_perceptual_v4_2_5_d_window(gt_rTG, pred_rTG, dim=dim_this, is_norm=False, lpips_extend_batch_z=config.get('lpips_extend_batch_z'), lpips_extend_batch_xy=config.get('lpips_extend_batch_xy'), window_size_z=config.get('window_size_z'), window_size_xy=config.get('window_size_xy'))
                        else:
                            assert 0

                        rTG_lpips_loss_ave.add(rTG_lpips_loss.item())
                        loss = loss + rTG_lpips_loss
            else:
                rTG_loss = 0
                rTG_loss_ave.add(0)

        if config.get('loss_optical_flow_tg') is not None:
            if epoch > config.get('loss_optical_flow_tg_epoch'):



                loss_optical_flow_tg_weight = config.get('loss_optical_flow_tg') * config.get('loss_lpips') * lpips_loss_weight_array[epoch - 1]

                gt_optical_flow = gt[:, 1:, :, :, :] - gt[:, :-1, :, :, :]
                pred_optical_flow = pred[:, 1:, :, :, :] - pred[:, :-1, :, :, :]
                gt_optical_flow = gt_optical_flow.reshape(1, -1, *gt_optical_flow.shape[3:])
                pred_optical_flow = pred_optical_flow.reshape(1, -1, *pred_optical_flow.shape[3:])




                dim_this = loss_perceptual_dim_list[epoch - 1]
                optical_flow_tg_loss = loss_optical_flow_tg_weight * loss_perceptual_v4_2_5_d_window(gt_optical_flow, pred_optical_flow, dim=dim_this, is_norm=False, lpips_extend_batch_z=config.get('lpips_extend_batch_z'), lpips_extend_batch_xy=config.get('lpips_extend_batch_xy'), window_size_z=config.get('window_size_z'), window_size_xy=config.get('window_size_xy'))
                optical_flow_tg_loss_ave.add(optical_flow_tg_loss.item())
                loss = loss + optical_flow_tg_loss
            else:
                optical_flow_tg_loss = 0
                optical_flow_tg_loss_ave.add(0)



        if config.get('loss_seq_select_num') is not None:
            global seq_selection_weight
            selected_index = torch.multinomial(seq_selection_weight.view(-1), config.get('loss_seq_select_num'), replacement=False, out=None)



            gt = gt[:, selected_index, :, :, :]
            pred = pred[:, selected_index, :, :, :]

        if config.get('gt_merge_seq'):
            pred = pred.reshape(1, -1, *pred.shape[3:])
            gt = gt.reshape(1, -1, *gt.shape[3:])

        if config.get('loss_tv_depth') is not None:
            tv_depth_loss = config.get('loss_tv_depth') * get_tv_depth_loss(pred)
            tv_depth_loss_ave.add(tv_depth_loss.item())
            loss = loss + tv_depth_loss

        if config.get('loss_l2') is not None:
            if config.get('log_detail_loss'):
                l2_loss = 0
                for seq_id in range(len(selected_index)):
                    l2_loss_this = config.get('loss_l2') * L2_pixelwise(pred[:,seq_id], gt[:,seq_id])
                    l2_loss = l2_loss + l2_loss_this
                    if config.get('record_each_seq') is not None:
                        record_each_seq[selected_index[seq_id]].add(l2_loss_this.item())
            else:
                l2_loss = config.get('loss_l2') * L2_pixelwise(pred, gt)
            l2_loss_ave.add(l2_loss.item())
            loss = loss + l2_loss
        if config.get('loss_lpips') is not None:
            lpips_loss_weight = config.get('loss_lpips')
            if config.get('loss_lpips_indices') is not None:


                lpips_loss_weight = lpips_loss_weight * lpips_loss_weight_array[epoch - 1]
            if lpips_loss_weight == 0:
                lpips_loss = 0
                lpips_loss_ave.add(0)
            else:

                if config.get('loss_perceptual_2_5_d') is not None:

                    dim_this = loss_perceptual_dim_list[epoch - 1]
                    if config.get('loss_perceptual_2_5_d_norm') is not None:
                        loss_perceptual_2_5_d_norm = config.get('loss_perceptual_2_5_d_norm')
                    else:
                        loss_perceptual_2_5_d_norm = False

                    if config.get('loss_perceptual_2_5_d_interpolate') is not None:
                        loss_perceptual_2_5_d_interpolate = config.get('loss_perceptual_2_5_d_interpolate')
                    else:
                        loss_perceptual_2_5_d_interpolate = False


                    if config.get('loss_perceptual_v2') is not None:
                        lpips_loss = lpips_loss_weight * loss_perceptual_v2(gt, pred, is_norm=loss_perceptual_2_5_d_norm)
                    elif config.get('loss_perceptual_v3') is not None:
                        loss_perceptual_v3_is_mean = False
                        if config.get('loss_perceptual_v3_is_mean') is not None:
                            loss_perceptual_v3_is_mean = config.get('loss_perceptual_v3_is_mean')
                        lpips_loss = lpips_loss_weight * loss_perceptual_v3(gt, pred, is_norm=loss_perceptual_2_5_d_norm, is_mean=loss_perceptual_v3_is_mean, extend_batch=config.get('lpips_extend_batch'))
                    elif config.get('loss_perceptual_v4') is not None:
                        lpips_loss = lpips_loss_weight * loss_perceptual_v4(gt, pred, is_norm=loss_perceptual_2_5_d_norm)
                    elif config.get('loss_perceptual_v4_2_5_d') is not None:
                        lpips_loss = lpips_loss_weight * loss_perceptual_v4_2_5_d(gt, pred, dim=dim_this, is_norm=loss_perceptual_2_5_d_norm, lpips_extend_batch_z=config.get('lpips_extend_batch_z'), lpips_extend_batch_xy=config.get('lpips_extend_batch_xy'))
                    elif config.get('loss_perceptual_v4_2_5_d_window') is not None:
                        if config.get('loss_perceptual_v4_2_5_d_window_all_dim_epoch') is not None and epoch > config.get('loss_perceptual_v4_2_5_d_window_all_dim_epoch'):
                            lpips_loss = 0
                            for _dim_this in range(1, 4):
                                lpips_loss = lpips_loss + lpips_loss_weight * loss_perceptual_v4_2_5_d_window(gt, pred, dim=_dim_this, is_norm=loss_perceptual_2_5_d_norm, lpips_extend_batch_z=config.get('lpips_extend_batch_z'), lpips_extend_batch_xy=config.get('lpips_extend_batch_xy'), window_size_z=config.get('window_size_z'), window_size_xy=config.get('window_size_xy'))
                            lpips_loss = lpips_loss / 3
                        else:
                            lpips_loss = lpips_loss_weight * loss_perceptual_v4_2_5_d_window(gt, pred, dim=dim_this, is_norm=loss_perceptual_2_5_d_norm, lpips_extend_batch_z=config.get('lpips_extend_batch_z'), lpips_extend_batch_xy=config.get('lpips_extend_batch_xy'), window_size_z=config.get('window_size_z'), window_size_xy=config.get('window_size_xy'))
                    else:
                        if config.get('log_detail_loss'):
                            lpips_loss = 0
                            for seq_id in range(len(selected_index)):
                                lpips_loss_this = lpips_loss_weight * loss_perceptual_2_5_d(gt[:,seq_id], pred[:,seq_id], dim=dim_this, is_norm=loss_perceptual_2_5_d_norm, is_interpolate=loss_perceptual_2_5_d_interpolate)
                                if config.get('record_each_seq') is not None:
                                    record_each_seq[selected_index[seq_id]].add(lpips_loss_this.item())
                                lpips_loss = lpips_loss + lpips_loss_this
                        else:
                            lpips_loss = lpips_loss_weight * loss_perceptual_2_5_d(gt, pred, dim=dim_this, is_norm=loss_perceptual_2_5_d_norm, is_interpolate=loss_perceptual_2_5_d_interpolate)
                else:
                    pass
                lpips_loss_ave.add(lpips_loss.item())
            loss = loss + lpips_loss



        if config.get('loss_l2_segment') is not None:
            if config.get('log_detail_loss'):
                l2_loss_seg = 0
                for seq_id in range(len(selected_index)):
                    pred_this = pred[:,seq_id]
                    gt_this = gt[:,seq_id]
                    if epoch < config.get('loss_l2_segment_epoch'):
                        l2_loss_seg_this = config.get('loss_l2_segment') * L2_pixelwise(pred_this[gt_this < 0.1], gt_this[gt_this < 0.1]) + config.get('loss_l2_segment') * L2_pixelwise(pred_this[gt_this >= 0.1], gt_this[gt_this >= 0.1])
                    else:
                        l2_loss_seg_this = config.get('loss_l2_segment') * L2_pixelwise(pred_this, gt_this)

                    l2_loss_seg = l2_loss_seg + l2_loss_seg_this
                    if config.get('record_each_seq') is not None:
                        record_each_seq[selected_index[seq_id]].add(l2_loss_seg_this.item())
            else:
                if epoch < config.get('loss_l2_segment_epoch'):
                    l2_loss_seg = config.get('loss_l2_segment') * L2_pixelwise(pred[gt < 0.1], gt[gt < 0.1]) + config.get('loss_l2_segment') * L2_pixelwise(pred[gt >= 0.1], gt[gt >= 0.1])
                else:
                    l2_loss_seg = config.get('loss_l2_segment') * L2_pixelwise(pred, gt)

            l2_segment_loss_ave.add(l2_loss_seg.item())
            loss = loss + l2_loss_seg

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None
        inp = None

    if config.get('loss_l2') is not None:
        log_info.append('l2_loss={:.4f}, '.format(l2_loss_ave.item()))

    if config.get('loss_l1') is not None:
        log_info.append('l1_loss={:.4f}, '.format(l1_loss_ave.item()))

    if config.get('loss_gradient') is not None:
        log_info.append('gradient_loss={:.4f}, '.format(gradient_loss_ave.item()))

    if config.get('loss_ssim') is not None:
        log_info.append('ssim_loss={:.4f}, '.format(ssim_loss_ave.item()))

    if config.get('loss_lpips') is not None:
        log_info.append('lpips_loss={:.4f}, '.format(lpips_loss_ave.item()))


    if config.get('loss_tv') is not None:
        log_info.append('tv_loss={:.4f}, '.format(tv_loss_ave.item()))
    if config.get('loss_rTG') is not None:
        log_info.append('rTG_loss={:.4f}, '.format(rTG_loss_ave.item()))

    if config.get('loss_l2_segment') is not None:
        log_info.append('l2_segment_loss={:.4f}, '.format(l2_segment_loss_ave.item()))

    if config.get('loss_optical_flow_tg') is not None:
        log_info.append('optical_flow_tg_loss={:.4f}, '.format(optical_flow_tg_loss_ave.item()))

    if config.get('loss_rTG_lpips') is not None:
        log_info.append('rTG_lpips_loss={:.4f}, '.format(rTG_lpips_loss_ave.item()))

    if config.get('loss_tv_depth') is not None:
        log_info.append('tv_depth_loss={:.4f}, '.format(tv_depth_loss_ave.item()))

    if config.get('log_detail_loss'):
        for seq_id in range(config.get('seq_len')):
            record_each_seq_this = record_each_seq[seq_id]
            writer.add_scalars(f'loss_seq_{seq_id}', {f'loss_seq_{seq_id}': record_each_seq_this.item()}, epoch)

    check_model_params_update(model)
    return train_loss.item()


def main(config_, save_path):
    global config, log, writer, epoch, threshold, epoch_threshold, log_info, epoch_max
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    snapshot_code_directory(save_path)

    train_loader, val_loader = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18
    threshold = config.get('threshold')
    epoch_threshold = config.get('epoch_threshold')

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        sv_file = build_checkpoint(config, model, optimizer, epoch)

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            val_res = eval_psnr_seq4(
                val_loader,
                model,
                writer=writer,
                config=config,
                EPOCH=epoch,
                seq_len=config.get('seq_len'),
            )

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--resume', default=None, help='Path to checkpoint file to resume training from')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    initialize_runtime_from_config(config)

    save_name = build_save_name(args)
    save_path = os.path.join('../save', save_name)

    main(config, save_path)
