import os
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import re
import sys
import glob
import argparse
import logging
import imageio
import torch
import numpy as np
import time
import math
from tifffile import imwrite
import torch.nn.functional as F

# ANSI styling is used only for terminal output; log files remain plain text.
_C_RESET = "\033[0m"
_C_DIM = "\033[2m"
_C_CYAN = "\033[36m"
_C_YELLOW = "\033[93m"
_RE_TIMING_LINE = re.compile(r"^(?P<prefix>.*?:\s*)(?P<num>\d+\.\d+)s$")

_TIMING_COLOR_ENABLED = False


def set_timing_color_enabled(enabled: bool) -> None:
    global _TIMING_COLOR_ENABLED
    _TIMING_COLOR_ENABLED = bool(enabled)


def style_timing_message(plain: str) -> str:
    """Dim timing labels and highlight durations in terminal output."""
    if not _TIMING_COLOR_ENABLED:
        return plain
    m = _RE_TIMING_LINE.match(plain.strip())
    if not m:
        return f"{_C_CYAN}{plain}{_C_RESET}"
    return f"{_C_DIM}{m.group('prefix')}{_C_RESET}{_C_YELLOW}{m.group('num')}s{_C_RESET}"


def log_timing(obj, filename="log.txt"):
    """Print timing text with optional color and write plain text to the log."""
    import utils as _utils

    plain = obj if isinstance(obj, str) else str(obj)
    out = style_timing_message(plain) if _TIMING_COLOR_ENABLED else plain
    print(out)
    if _utils._log_path is not None:
        with open(os.path.join(_utils._log_path, filename), "a") as f:
            print(plain, file=f)


def get_all_abs_path(source_dir):
    path_list = []
    for fpathe, dirs, fs in os.walk(source_dir):
        for f in fs:
            p = os.path.join(fpathe, f)
            path_list.append(p)
    return path_list


def preDAO_v2(lfstack, defocus=0, max_iter=10, log_func=None):
    """Register the 13 angular views by iterative phase correlation."""
    if lfstack.ndim != 3 or lfstack.shape[0] != 13:
        raise ValueError(
            f"preDAO_v2 expects a [13, H, W] tensor, got {tuple(lfstack.shape)}"
        )
    if max_iter < 0:
        raise ValueError("preDAO_max_iter must be non-negative")

    reference_indices = torch.tensor(
        [
            [0, 0, 1, 1, 12, 11, 11, 10, 10, 1, 0, 0, 0],
            [0, 0, 1, 12, 12, 12, 11, 11, 10, 10, 0, 0, 0],
        ],
        dtype=torch.long,
        device=lfstack.device,
    )
    view_i = lfstack.new_tensor(
        [0, 0, 0, -77, 96, -77, 0, 77, 96, 77, 52, 0, -52]
    )
    view_j = lfstack.new_tensor(
        [0, 52, 96, 77, 0, -77, -96, -77, 0, 77, 0, -52, 0]
    )
    identity = lfstack.new_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    lf_grid = F.affine_grid(
        identity.unsqueeze(0),
        size=(1, lfstack.shape[0], lfstack.shape[1], lfstack.shape[2]),
        align_corners=False,
    ).repeat(lfstack.shape[0], 1, 1, 1)
    shift_lf_sum = lfstack.new_zeros((lfstack.shape[0], 1, 1, 2))

    with torch.no_grad():
        for _ in range(max_iter):
            ref = F.interpolate(
                lfstack.unsqueeze(1),
                size=(lfstack.shape[-2] * 2 - 1, lfstack.shape[-1] * 2 - 1),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

            shift_lf_x = lfstack.new_zeros(lfstack.shape[0])
            shift_lf_y = lfstack.new_zeros(lfstack.shape[0])
            for indices in reference_indices:
                pred_lf = ref[indices]
                corr = torch.fft.fftshift(
                    torch.fft.ifft2(
                        torch.fft.fft2(
                            torch.fft.ifftshift(pred_lf, dim=(-2, -1))
                        )
                        * torch.fft.fft2(
                            torch.fft.ifftshift(
                                ref.flip((-2, -1)),
                                dim=(-2, -1),
                            )
                        )
                    ),
                    dim=(-2, -1),
                ).real
                shift_lf_x += (
                    pred_lf.shape[-1] // 2
                    - corr.amax(dim=-2).argmax(dim=-1)
                )
                shift_lf_y += (
                    pred_lf.shape[-2] // 2
                    - corr.amax(dim=-1).argmax(dim=-1)
                )

            shift_lf_x /= reference_indices.shape[0]
            shift_lf_y /= reference_indices.shape[0]
            if defocus:
                denominator = (view_i.square() + view_j.square()).sum()
                k = (view_i * shift_lf_y + view_j * shift_lf_x).sum() / denominator
                shift_lf_y -= k * view_i
                shift_lf_x -= k * view_j

            shift_lf = torch.stack((shift_lf_x, shift_lf_y), dim=-1)
            shift_lf = shift_lf[:, None, None, :] - shift_lf[0, None, None, :]
            shift_lf_sum += shift_lf / 2
            normalized_shift = shift_lf * 2 / ref.shape[-1]
            lfstack = F.grid_sample(
                lfstack.unsqueeze(1),
                lf_grid + normalized_shift,
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

    shift_display = shift_lf_sum.squeeze(1).squeeze(1).T.cpu().numpy()
    if log_func:
        log_func(f"preDAO_v2 total shift after {max_iter} iterations:")
        log_func(
            f"X: {str(shift_display[0].round(2)).replace('[', '').replace(']', '')}"
        )
        log_func(
            f"Y: {str(shift_display[1].round(2)).replace('[', '').replace(']', '')}"
        )
    return lfstack, shift_lf_sum


def _patch_starts(length, patch_size, overlap):
    if patch_size >= length:
        return [0]
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than inp_size ({patch_size})"
        )
    return [
        i * stride
        for i in range(1 + math.ceil((length - patch_size) / stride))
    ]


def required_patch_hw_pairs(inp_size, overlap, height, width):
    """Return every spatial patch shape needed by the sliding-window loop."""
    return sorted(
        {
            (
                min(inp_size, height - h0),
                min(inp_size, width - w0),
            )
            for h0 in _patch_starts(height, inp_size, overlap)
            for w0 in _patch_starts(width, inp_size, overlap)
        }
    )


class TRTInference:
    """Load static TensorRT engines and select one by input patch size."""

    def __init__(self, engine_paths, use_cuda_graph=False):
        import tensorrt as trt

        self.trt = trt
        self.use_cuda_graph = use_cuda_graph
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        # A TensorRT engine/context retains resources owned by its runtime.
        # Keep the full ownership chain alive for as long as this runner is in
        # use; releasing a local runtime can otherwise cause a native crash.
        self.runtimes = []
        self.engines = []
        self.contexts = []
        self.input_names = []
        self.output_names = []
        self.input_shapes = []
        self.input_dtypes = []
        self.output_dtypes = []
        self.cuda_graphs = []
        self.cuda_graph_streams = []
        self.graph_input_buffers = []
        self.graph_output_buffers = []
        self.shape_to_idx = {}

        for path in engine_paths:
            with open(path, "rb") as file_obj:
                runtime = trt.Runtime(self.logger)
                engine = runtime.deserialize_cuda_engine(file_obj.read())
            if engine is None:
                raise RuntimeError(f"[TRT] Failed to deserialize engine: {path}")

            input_names = [
                engine.get_tensor_name(i)
                for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i))
                == trt.TensorIOMode.INPUT
            ]
            output_names = [
                engine.get_tensor_name(i)
                for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i))
                == trt.TensorIOMode.OUTPUT
            ]
            if len(input_names) != 1 or len(output_names) != 1:
                raise ValueError(
                    f"[TRT] Expected one input and one output in {path}; "
                    f"found inputs={input_names}, outputs={output_names}"
                )

            input_name = input_names[0]
            output_name = output_names[0]
            input_shape = tuple(engine.get_tensor_shape(input_name))
            if len(input_shape) not in (4, 5) or any(dim < 1 for dim in input_shape):
                raise ValueError(
                    f"[TRT] Expected a static NCHW/NSCHW input, got "
                    f"{input_shape} ({path})"
                )

            hw = (int(input_shape[-2]), int(input_shape[-1]))
            if hw in self.shape_to_idx:
                raise ValueError(
                    f"[TRT] Duplicate patch size {hw}: {path}"
                )

            self.shape_to_idx[hw] = len(self.engines)
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError(
                    f"[TRT] Failed to create execution context: {path}"
                )
            self.runtimes.append(runtime)
            self.engines.append(engine)
            self.contexts.append(context)
            self.input_names.append(input_name)
            self.output_names.append(output_name)
            self.input_shapes.append(input_shape)
            self.input_dtypes.append(self._torch_dtype(engine.get_tensor_dtype(input_name)))
            self.output_dtypes.append(self._torch_dtype(engine.get_tensor_dtype(output_name)))
            self.cuda_graphs.append(None)
            self.cuda_graph_streams.append(None)
            self.graph_input_buffers.append(None)
            self.graph_output_buffers.append(None)
            logging.info(
                "[TRT] Loaded %s: input %s %s, output %s %s",
                path,
                input_name,
                input_shape,
                output_name,
                tuple(engine.get_tensor_shape(output_name)),
            )

    def _torch_dtype(self, trt_dtype):
        np_dtype = np.dtype(self.trt.nptype(trt_dtype))
        mapping = {
            np.dtype(np.float16): torch.float16,
            np.dtype(np.float32): torch.float32,
            np.dtype(np.int8): torch.int8,
            np.dtype(np.int32): torch.int32,
            np.dtype(np.int64): torch.int64,
            np.dtype(np.bool_): torch.bool,
        }
        if np_dtype not in mapping:
            raise TypeError(f"[TRT] Unsupported tensor dtype: {trt_dtype}")
        return mapping[np_dtype]

    def _prepare_cuda_graph(self, input_tensor, engine_index):
        """Capture one fixed-address execution graph for a static engine."""
        context = self.contexts[engine_index]
        input_name = self.input_names[engine_index]
        output_name = self.output_names[engine_index]
        output_shape = tuple(context.get_tensor_shape(output_name))
        if any(dim < 1 for dim in output_shape):
            raise ValueError(f"[TRT] Invalid output shape: {output_shape}")

        input_buffer = torch.empty_like(input_tensor)
        output_buffer = torch.empty(
            output_shape,
            dtype=self.output_dtypes[engine_index],
            device=input_tensor.device,
        )
        context.set_tensor_address(input_name, input_buffer.data_ptr())
        context.set_tensor_address(output_name, output_buffer.data_ptr())

        current_stream = torch.cuda.current_stream(input_tensor.device)
        graph_stream = torch.cuda.Stream(device=input_tensor.device)
        input_buffer.copy_(input_tensor)
        graph_stream.wait_stream(current_stream)
        with torch.cuda.stream(graph_stream):
            for _ in range(2):
                if not context.execute_async_v3(graph_stream.cuda_stream):
                    raise RuntimeError(
                        f"[TRT] CUDA Graph warmup failed for engine {engine_index}"
                    )
        graph_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(graph_stream):
            with torch.cuda.graph(graph, stream=graph_stream):
                if not context.execute_async_v3(graph_stream.cuda_stream):
                    raise RuntimeError(
                        f"[TRT] CUDA Graph capture failed for engine {engine_index}"
                    )
        graph_stream.synchronize()

        self.graph_input_buffers[engine_index] = input_buffer
        self.graph_output_buffers[engine_index] = output_buffer
        self.cuda_graph_streams[engine_index] = graph_stream
        self.cuda_graphs[engine_index] = graph

    def infer(self, input_tensor: torch.Tensor, engine_index: int) -> torch.Tensor:
        if not input_tensor.is_cuda:
            raise ValueError("[TRT] Input tensor must be on CUDA")
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        engine = self.engines[engine_index]
        context = self.contexts[engine_index]
        expected_shape = self.input_shapes[engine_index]
        expected_dtype = self.input_dtypes[engine_index]
        if tuple(input_tensor.shape) != expected_shape:
            raise ValueError(
                f"[TRT] Engine {engine_index} expects {expected_shape}, "
                f"got {tuple(input_tensor.shape)}"
            )
        if input_tensor.dtype != expected_dtype:
            raise TypeError(
                f"[TRT] Engine {engine_index} expects {expected_dtype}, "
                f"got {input_tensor.dtype}"
            )

        if self.use_cuda_graph:
            if self.cuda_graphs[engine_index] is None:
                self._prepare_cuda_graph(input_tensor, engine_index)
            current_stream = torch.cuda.current_stream(input_tensor.device)
            graph_stream = self.cuda_graph_streams[engine_index]
            self.graph_input_buffers[engine_index].copy_(input_tensor)
            graph_stream.wait_stream(current_stream)
            with torch.cuda.stream(graph_stream):
                self.cuda_graphs[engine_index].replay()
            current_stream.wait_stream(graph_stream)
            return self.graph_output_buffers[engine_index]

        input_name = self.input_names[engine_index]
        output_name = self.output_names[engine_index]
        output_shape = tuple(context.get_tensor_shape(output_name))
        if any(dim < 1 for dim in output_shape):
            raise ValueError(f"[TRT] Invalid output shape: {output_shape}")
        output_tensor = torch.empty(
            output_shape,
            dtype=self.output_dtypes[engine_index],
            device=input_tensor.device,
        )

        context.set_tensor_address(input_name, input_tensor.data_ptr())
        context.set_tensor_address(output_name, output_tensor.data_ptr())
        stream = torch.cuda.current_stream(input_tensor.device).cuda_stream
        if not context.execute_async_v3(stream):
            raise RuntimeError(
                f"[TRT] execute_async_v3 failed for engine {engine_index}"
            )
        return output_tensor

    def infer_for_tensor(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hw = tuple(int(value) for value in input_tensor.shape[-2:])
        index = self.shape_to_idx.get(hw)
        if index is None:
            raise ValueError(
                f"[TRT] No engine for patch size {hw[0]}x{hw[1]}; "
                f"loaded sizes: {sorted(self.shape_to_idx)}"
            )
        return self.infer(input_tensor, index)


def build_trt_runner(trt_engine_dir: str, use_cuda_graph=False):
    """Load static engines and select one from the input spatial shape."""
    paths = sorted(glob.glob(os.path.join(trt_engine_dir, "*.engine")))
    if not paths:
        raise FileNotFoundError(f"[TRT] No .engine files found in {trt_engine_dir}")
    trt_wr = TRTInference(paths, use_cuda_graph=use_cuda_graph)

    def infer_fn(inp: torch.Tensor, _scale_tensor: torch.Tensor) -> torch.Tensor:
        return trt_wr.infer_for_tensor(inp)

    return infer_fn, trt_wr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', default="WAWA", type=str)
    parser.add_argument('--savefolder', default="WAWA", type=str)
    parser.add_argument('--model', default='WAWA', type=str)
    parser.add_argument('--resolution_z', default=60, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--order', default=1, type=int)
    parser.add_argument('--inp_size', default=237, type=int)
    parser.add_argument('--overlap', default=15, type=int)
    parser.add_argument('--startframe', default=0, type=int)
    parser.add_argument('--percentile_lowerbound', default=0, type=float)
    parser.add_argument('--percentile_upperbound', default=None, type=float)
    parser.add_argument('--codefolder', default="/code_not_found/", type=str)
    parser.add_argument('--replacestr', default='undefined', type=str)
    parser.add_argument('--save_file_name', default='trans', type=str)
    parser.add_argument('--begin_frame', default=1, type=int)
    parser.add_argument('--end_frame', default=None, type=int)
    parser.add_argument('--anglewise_normalization', default=1, type=int)
    parser.add_argument('--series_normalization', default=1, type=int)
    parser.add_argument('--save_tif', default=1, type=int)
    parser.add_argument('--save_MIPz', default=1, type=int)
    parser.add_argument('--save_individual', default=0, type=int)
    parser.add_argument('--MIPz_cutedge', default=5, type=int)
    parser.add_argument(
        '--preDAO',
        default='preDAO_off',
        choices=('preDAO_off', 'preDAO_v2'),
    )
    parser.add_argument('--preDAO_max_iter', default=5, type=int)
    parser.add_argument('--preDAO_defocus', default=0, type=int)
    parser.add_argument('--seq_len', default=4, type=int)
    parser.add_argument('--percentile_scale', default=1, type=float)
    parser.add_argument(
        '--backend',
        default='pytorch',
        type=str,
        choices=('pytorch', 'tensorrt'),
        help='pytorch loads --model; tensorrt selects static engines from --trt_engine_dir',
    )
    parser.add_argument(
        '--trt_engine_dir',
        default='',
        type=str,
        help='Directory containing one static TensorRT engine per required patch shape',
    )
    parser.add_argument(
        '--trt_cuda_graph',
        action='store_true',
        help='Capture a fixed-address CUDA Graph for each static TensorRT engine',
    )
    parser.add_argument(
        '--no_color',
        action='store_true',
        help='Disable ANSI timing colors (also controlled by NO_COLOR=1)',
    )
    return parser.parse_args()

def load_frame_data(file_path):
    """Load one light-field frame."""
    lfstack = torch.from_numpy(np.array(imageio.volread(file_path), dtype=np.float32))
    lfstack[lfstack.isnan()] = 0
    lfstack[lfstack.isinf()] = 0
    return lfstack

def process_batch(infer_fn, lfstack_batch, args, d, h, w, inp_size, overlap, weight, scale, seq_len, log_func=None):
    """Run one batch through ``infer_fn`` and return its CUDA prediction."""
    if seq_len > 1:
        inp_all = lfstack_batch.unsqueeze(0)
    else:
        inp_all = lfstack_batch
    device = lfstack_batch.device
    scale_tensor = torch.tensor([scale], dtype=torch.float32, device=device).unsqueeze(0)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    
    if inp_size >= lfstack_batch.shape[-2] and inp_size >= lfstack_batch.shape[-1]:
        with torch.no_grad():
            pred = infer_fn(((inp_all - 0) / 1), scale_tensor)
        pred[pred<0] = 0
        pred = pred.squeeze()
        ret = pred
        pred = None
    else:
        if log_func:
            log_func('patched and sigmoid-based image fusion for overlap')
        
        ret = torch.zeros([seq_len, d, h, w], device=device)
        base = torch.zeros_like(ret)
        weight = weight.to(device, non_blocking=True)
        for h0 in [i*(inp_size-overlap) for i in range(1+math.ceil((inp_all.shape[-2]-inp_size)/(inp_size-overlap)))]:
            for w0 in [i*(inp_size-overlap) for i in range(1+math.ceil((inp_all.shape[-1]-inp_size)/(inp_size-overlap)))]:
                if seq_len > 1:
                    inp = inp_all[:,:,:,h0:h0+inp_size,w0:w0+inp_size]
                else:
                    inp = inp_all[:,:,h0:h0+inp_size,w0:w0+inp_size]
                with torch.no_grad():
                    pred = infer_fn(((inp - 0) / 1), scale_tensor)
                pred.clamp_(min=0)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                ret[:,:,h0:(h0+inp_size),w0:(w0+inp_size)] = \
                    ret[:,:,h0:(h0+inp_size),w0:(w0+inp_size)] + pred * weight[:,:,0:pred.shape[-2],0:pred.shape[-1]]
                base[:,:,h0:(h0+inp_size),w0:(w0+inp_size)] = \
                    base[:,:,h0:(h0+inp_size),w0:(w0+inp_size)] + weight[:,:,0:pred.shape[-2],0:pred.shape[-1]]
                pred = None
        ret = ret / base.clamp_min_(1e-6)
    
    ender.record()
    torch.cuda.synchronize()
    elapsed_ms = starter.elapsed_time(ender)
    
    if log_func:
        log_func(f"inference time: {elapsed_ms / 1000:.4f}s")
    return ret

def save_results(ret, args, original_frame_ids, seq_len, save_tif, save_MIPz, save_individual, MIPz_cutedge, series_normalization, MIPz_series=None, log_func=None):
    """Save reconstructed volumes and optional MIPz images."""
    print('ret shape',ret.shape)
    tifffileimwritecompression = 'zlib'
    if save_tif == 1:
        t0 = time.perf_counter()
        coef = 30000
        if seq_len == 1:
            imwrite(os.path.join(args.savefolder, f"{args.save_file_name}_{original_frame_ids[0]}.tif"), np.uint16(ret.squeeze() * coef), imagej=True, metadata={'axes': 'ZYX'}, compression=tifffileimwritecompression)
        else:
            for seq_id in range(seq_len):
                imwrite(os.path.join(args.savefolder, f"{args.save_file_name}_{original_frame_ids[seq_id]}.tif"), np.uint16(ret[seq_id] * coef), imagej=True, metadata={'axes': 'ZYX'}, compression=tifffileimwritecompression)
        if log_func:
            log_func(f"save_results write tif: {time.perf_counter() - t0:.4f}s")

    if save_MIPz == 1:
        if seq_len == 1:
            ret_squeezed = ret.squeeze(0)
            ret_select = ret_squeezed[MIPz_cutedge: -MIPz_cutedge]
            print('ret_select shape',ret_select.shape)
            MIPz, _ = torch.max(ret_select, axis=0, keepdim=True)
            if series_normalization and MIPz_series is not None:
                global_frame_index = original_frame_ids[0] - args.begin_frame
                MIPz_series[global_frame_index,:,:] = MIPz
            if save_individual == 1:
                coef = 30000
                imwrite(os.path.join(args.savefolder, f"MIPz_{args.save_file_name}_{original_frame_ids[0]}.tif"), np.uint16(MIPz * coef), imagej=True, metadata={'axes': 'ZYX'}, compression=tifffileimwritecompression)
        else:  
            for seq_id in range(seq_len):
                ret_select = ret[seq_id, MIPz_cutedge: -MIPz_cutedge]
                MIPz, _ = torch.max(ret_select, axis=0, keepdim=True)
                if series_normalization and MIPz_series is not None:
                    global_frame_index = original_frame_ids[seq_id] - args.begin_frame
                    MIPz_series[global_frame_index,:,:] = MIPz
                if save_individual == 1:
                    coef = 30000
                    imwrite(os.path.join(args.savefolder, f"MIPz_{args.save_file_name}_{original_frame_ids[seq_id]}.tif"), np.uint16(MIPz * coef), imagej=True, metadata={'axes': 'ZYX'}, compression=tifffileimwritecompression)

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("\n" + "="*80)
    print("Configuration Arguments:")
    print("="*80)
    for arg_name, arg_value in vars(args).items():
        print(f"  {arg_name:25s}: {arg_value}")
    print("="*80 + "\n")
    d = args.resolution_z
    save_MIPz = args.save_MIPz
    save_tif = args.save_tif
    save_individual = args.save_individual
    MIPz_cutedge = args.MIPz_cutedge

    args.savefolder = args.savefolder
    os.makedirs(args.savefolder, exist_ok=True)

    sys.path.append(args.codefolder)

    import utils

    utils.set_log_path(args.savefolder[0:args.savefolder.find(args.replacestr)] + args.replacestr)
    log = utils.log
    use_timing_color = (
        sys.stdout.isatty()
        and not os.environ.get("NO_COLOR")
        and not args.no_color
    )
    set_timing_color_enabled(use_timing_color)
    log(f"sys.path: {sys.path}")
    log(f'using codefolder {args.codefolder}')

    if args.backend == 'pytorch':
        import models

        model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
        model.eval()
        log(f'using model {args.model}')

        def infer_fn(inp, scale_tensor):
            return model(inp, scale_tensor)
    else:
        if not args.trt_engine_dir:
            raise ValueError("--trt_engine_dir is required for backend=tensorrt")
        infer_fn, trt_wr = build_trt_runner(
            args.trt_engine_dir,
            use_cuda_graph=args.trt_cuda_graph,
        )
        log(f'using TensorRT engines from {args.trt_engine_dir}')
        log(f'loaded patch sizes (H,W): {sorted(trt_wr.shape_to_idx.keys())}')
        log(f'TensorRT CUDA Graph: {args.trt_cuda_graph}')

    log(args.savefolder.replace(args.replacestr,''))

    t_scan = time.perf_counter()
    files = get_all_abs_path(args.datapath)
    files = [file for file in files if file.endswith('.tif')]
    if args.order:
        files = sorted(files, key=lambda x: int(x[x.rfind(os.sep)+5:-4]))
    else:
        files = sorted(files)
    
    begin_frame = args.begin_frame
    end_frame = args.end_frame if args.end_frame is not None else len(files)
    total_frames = end_frame - begin_frame + 1
    
    log(f"Processing frames from {begin_frame} to {end_frame}, total {total_frames} frames")
    
    files = files[begin_frame-1:end_frame]
    log_timing(f"file scan, filter, sort & range select: {time.perf_counter() - t_scan:.4f}s")
    
    t_first = time.perf_counter()
    first_frame = load_frame_data(files[0])
    log_timing(f"load first frame (shape metadata): {time.perf_counter() - t_first:.4f}s")
    h, w = first_frame.shape[1], first_frame.shape[2]

    scale = 1
    inp_size = min(args.inp_size, h, w)
    overlap = args.overlap

    if args.backend == 'tensorrt':
        need_pairs = required_patch_hw_pairs(inp_size, overlap, h, w)
        missing = [pair for pair in need_pairs if pair not in trt_wr.shape_to_idx]
        if missing:
            raise RuntimeError(
                f"inp_size={inp_size}, overlap={overlap}, and image size {h}x{w} "
                f"require patch shapes {need_pairs}, but engines are missing for {missing}."
            )
        log(f"TensorRT patch check passed with {len(need_pairs)} shape(s): {need_pairs}")

    overlapVolume = round(overlap * scale)
    if overlap:
        edge = torch.sigmoid((torch.arange(overlapVolume) - overlapVolume//2 ) / overlapVolume * 15)
        weight = torch.cat([edge, edge.max()*torch.ones(round(inp_size*scale) - 2*len(edge)), edge.flip(0)], dim=0).view(-1,1) @ \
            torch.cat([edge, edge.max()*torch.ones(round(inp_size*scale) - 2*len(edge)), edge.flip(0)], dim=0).view(1,-1) + 1e-3
        weight = weight.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
    else:
        weight = torch.ones(round(inp_size*scale), round(inp_size*scale)).unsqueeze(0)

    series_normalization = args.series_normalization
    seq_len = args.seq_len
    
    MIPz_series = None
    if save_MIPz == 1 and series_normalization == 1:
        MIPz_series = torch.zeros([total_frames, first_frame.shape[1], first_frame.shape[2]])

    for batch_start in range(0, len(files), seq_len):
        batch_end = min(batch_start + seq_len, len(files))
        current_seq_len = batch_end - batch_start
        
        batch_files = files[batch_start:batch_end]
        original_frame_ids = []
        for file_path in batch_files:
            frame_id = int(file_path[file_path.rfind(os.sep)+5:-4])
            original_frame_ids.append(frame_id)
        
        log(f"Processing batch {batch_start//seq_len + 1}: frames {original_frame_ids}")
        
        lfstack_batch = torch.zeros([current_seq_len, first_frame.shape[0], first_frame.shape[1], first_frame.shape[2]])
        
        t_load = time.perf_counter()
        for i, file_path in enumerate(batch_files):
            frame_data = load_frame_data(file_path)
            if args.preDAO == "preDAO_v2":
                frame_data, _ = preDAO_v2(
                    frame_data.cuda(),
                    defocus=args.preDAO_defocus,
                    max_iter=args.preDAO_max_iter,
                    log_func=log,
                )
                frame_data = frame_data.cpu()
            lfstack_batch[i] = frame_data
        log_timing(f"batch load & preDAO: {time.perf_counter() - t_load:.4f}s")
        
        t_norm = time.perf_counter()
        if series_normalization == 1:
            if args.anglewise_normalization == 1:
                for z in range(lfstack_batch.shape[1]):
                    lfstack_slice = lfstack_batch[:,z,:,:]
                    lfstack_high = np.percentile(lfstack_slice, args.percentile_upperbound) if args.percentile_upperbound is not None else 65535
                    lfstack_low = 0
                    lfstack_slice = ((lfstack_slice-lfstack_low) / (lfstack_high-lfstack_low) * args.percentile_scale)
                    lfstack_slice = lfstack_slice.clamp(0,1)
                    lfstack_batch[:,z,:,:] = lfstack_slice
            else:
                lfstack_high = np.percentile(lfstack_batch, args.percentile_upperbound) if args.percentile_upperbound is not None else 65535
                lfstack_low = np.percentile(lfstack_batch, args.percentile_lowerbound)
                lfstack_batch = ((lfstack_batch-lfstack_low) / (lfstack_high-lfstack_low) * args.percentile_scale)
                lfstack_batch = lfstack_batch.clamp(0,1)
        log_timing(f"batch percentile normalize: {time.perf_counter() - t_norm:.4f}s")
        
        lfstack_batch = lfstack_batch.cuda(non_blocking=True)
        ret = process_batch(
            infer_fn,
            lfstack_batch,
            args,
            d,
            h,
            w,
            inp_size,
            overlap,
            weight,
            scale,
            current_seq_len,
            log_func=log_timing,
        )
        ret = ret.cpu()
        save_results(
            ret,
            args,
            original_frame_ids,
            current_seq_len,
            save_tif,
            save_MIPz,
            save_individual,
            MIPz_cutedge,
            series_normalization,
            MIPz_series,
            log_func=log_timing,
        )
    
    if save_MIPz == 1 and series_normalization == 1:
        t_mipz_all = time.perf_counter()
        tifffileimwritecompression = 'zlib'
        coef = 30000
        imwrite(os.path.join(args.savefolder, f"MIPz_{args.save_file_name}_all.tif"), np.uint16(MIPz_series * coef), imagej=True, metadata={'axes': 'ZYX'}, compression=tifffileimwritecompression)
        log_timing(f"write MIPz series all tif: {time.perf_counter() - t_mipz_all:.4f}s")

if __name__ == '__main__':
    main()
