#!/usr/bin/env python3
"""Convert a fixed-shape STaR-Net checkpoint to ONNX and TensorRT.

Official Mamba2 CUDA/Triton operators are kept for training and PyTorch
inference. For ONNX export only, this script replaces them in memory with an
equivalent recurrent scan composed of standard PyTorch operators. The original
checkpoint is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--H", type=int, default=200, help="Static input height.")
    parser.add_argument("--W", type=int, default=200, help="Static input width.")
    parser.add_argument("--seq-len", type=int, default=4, dest="seq_len")
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument(
        "--model-dir-root",
        type=Path,
        default=Path("../save/train_starnet"),
        help="Directory containing epoch-*.pth.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help=(
            "Logical CUDA device. When CUDA_VISIBLE_DEVICES is set, this is an "
            "index within the visible devices."
        ),
    )
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument(
        "--optimization-level", type=int, choices=range(0, 6), default=3
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Export and fold ONNX, but do not build or validate an engine.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild files that already exist."
    )
    return parser.parse_args()


def input_shape(args):
    if args.seq_len == 1:
        return (1, 13, args.H, args.W)
    return (1, args.seq_len, 13, args.H, args.W)


def find_polygraphy():
    executable = shutil.which("polygraphy")
    if executable is None:
        candidate = Path(sys.executable).resolve().parent / "polygraphy"
        if candidate.is_file():
            executable = str(candidate)
    if executable is None:
        raise FileNotFoundError(
            "polygraphy is required for ONNX constant folding. Install it in "
            "the active TensorRT environment."
        )
    return executable


def torch_dtype(torch_mod, trt_mod, dtype):
    mapping = {
        trt_mod.float32: torch_mod.float32,
        trt_mod.float16: torch_mod.float16,
        trt_mod.int32: torch_mod.int32,
        trt_mod.int8: torch_mod.int8,
        trt_mod.bool: torch_mod.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


def write_report(path, report):
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote conversion report: {path.resolve()}")


def main():
    args = parse_args()
    if args.H <= 0 or args.W <= 0 or args.seq_len <= 0:
        raise ValueError("H, W, and seq-len must be positive")
    if args.workspace_gb <= 0:
        raise ValueError("workspace-gb must be positive")

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    import tensorrt as trt
    import torch
    from torch.onnx import TrainingMode

    import models
    from mamba2_exportable import replace_mamba2_for_export

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT conversion and validation")

    if args.gpu is not None:
        cuda_device_index = args.gpu
    else:
        cuda_device_index = 0
    device = torch.device(f"cuda:{cuda_device_index}")
    torch.cuda.set_device(device)
    print(f"Using {device}: {torch.cuda.get_device_name(device)}")

    model_dir = args.model_dir_root.resolve()
    checkpoint_path = model_dir / f"epoch-{args.epoch}.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint_path}")

    output_dir = model_dir / f"tensorrt_models_epoch{args.epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"model_H{args.H}_W{args.W}"
    raw_onnx = output_dir / f"{stem}.onnx"
    folded_onnx = output_dir / f"{stem}_folded.onnx"
    engine_path = output_dir / f"{stem}.engine"
    report_path = output_dir / f"{stem}_conversion_report.json"

    report = {
        "checkpoint": str(checkpoint_path),
        "input_shape": list(input_shape(args)),
        "opset": args.opset,
        "optimization_level": args.optimization_level,
        "workspace_gb": args.workspace_gb,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tensorrt": trt.__version__,
    }

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    export_model = models.make(checkpoint["model"], load_sd=True).to(device).eval()
    torch.manual_seed(1)
    dummy_input = torch.randn(
        input_shape(args), device=device, dtype=torch.float32
    )
    dummy_scale = torch.tensor([[1.0]], device=device, dtype=torch.float32)

    print("Checking export-only Mamba2 replacement against the native model...")
    with torch.inference_mode():
        native_output = export_model(dummy_input, dummy_scale).detach().clone()
    replaced = replace_mamba2_for_export(export_model)
    with torch.inference_mode():
        export_output = export_model(dummy_input, dummy_scale).detach().clone()
    export_diff = (native_output.float() - export_output.float()).abs()
    report.update(
        {
            "mamba2_modules_replaced": replaced,
            "export_fallback_max_abs_error": float(export_diff.max().item()),
            "export_fallback_mean_abs_error": float(export_diff.mean().item()),
            "export_fallback_allclose_atol_rtol_2e-3": bool(
                torch.allclose(native_output, export_output, atol=2e-3, rtol=2e-3)
            ),
            "output_shape": list(native_output.shape),
        }
    )
    print(
        f"Replaced {replaced} Mamba2 module(s); "
        f"max abs error={report['export_fallback_max_abs_error']:.6g}, "
        f"allclose={report['export_fallback_allclose_atol_rtol_2e-3']}"
    )
    if not report["export_fallback_allclose_atol_rtol_2e-3"]:
        write_report(report_path, report)
        raise RuntimeError(
            "Exportable Mamba2 failed the atol=rtol=2e-3 correctness check"
        )
    del native_output, export_output, export_diff

    if args.force or not raw_onnx.exists():
        print(f"Exporting fixed-shape ONNX: {raw_onnx}")
        start = time.perf_counter()
        torch.onnx.export(
            export_model,
            (dummy_input, dummy_scale),
            str(raw_onnx),
            input_names=["input", "scale"],
            output_names=["output"],
            opset_version=args.opset,
            dynamic_axes=None,
            training=TrainingMode.EVAL,
            do_constant_folding=True,
        )
        report["export_seconds"] = time.perf_counter() - start
        report["export_status"] = "completed"
    else:
        report["export_status"] = "skipped_existing"
    report["raw_onnx"] = str(raw_onnx.resolve())
    report["raw_onnx_bytes"] = raw_onnx.stat().st_size
    del export_model
    torch.cuda.empty_cache()

    if args.force or not folded_onnx.exists():
        print(f"Folding ONNX constants: {folded_onnx}")
        start = time.perf_counter()
        subprocess.run(
            [
                find_polygraphy(),
                "surgeon",
                "sanitize",
                str(raw_onnx),
                "--fold-constants",
                "--output",
                str(folded_onnx),
            ],
            check=True,
        )
        report["fold_seconds"] = time.perf_counter() - start
        report["fold_status"] = "completed"
    else:
        report["fold_status"] = "skipped_existing"
    report["folded_onnx"] = str(folded_onnx.resolve())
    report["folded_onnx_bytes"] = folded_onnx.stat().st_size

    if args.onnx_only:
        report["build_status"] = "skipped_onnx_only"
        write_report(report_path, report)
        return

    if args.force or not engine_path.exists():
        print(f"Building fixed-shape FP16 TensorRT engine: {engine_path}")
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        with folded_onnx.open("rb") as handle:
            parsed = parser.parse(handle.read())
        report["parsed_layers"] = network.num_layers
        if not parsed:
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            report["parser_errors"] = errors
            write_report(report_path, report)
            raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
        if network.num_inputs != 1 or network.num_outputs != 1:
            raise ValueError(
                "Expected the exported network to have one input and one output; "
                f"found {network.num_inputs} and {network.num_outputs}"
            )

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1024**3))
        )
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This TensorRT platform does not support fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
        if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.builder_optimization_level = args.optimization_level
        trt.init_libnvinfer_plugins(logger, "")
        start = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        report["build_seconds"] = time.perf_counter() - start
        if serialized is None:
            write_report(report_path, report)
            raise RuntimeError("TensorRT engine build failed")
        engine_path.write_bytes(serialized)
        report["build_status"] = "completed"
    else:
        report["build_status"] = "skipped_existing"
    report["engine"] = str(engine_path.resolve())
    report["engine_bytes"] = engine_path.stat().st_size

    print("Validating TensorRT output against a fresh native PyTorch model...")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create the TensorRT execution context")
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_names = [
        name
        for name in names
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    ]
    output_names = [
        name
        for name in names
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
    ]
    if len(input_names) != 1 or len(output_names) != 1:
        raise ValueError(
            f"Expected one input and one output, got {input_names} and {output_names}"
        )
    input_name, output_name = input_names[0], output_names[0]
    expected_input_shape = tuple(int(v) for v in engine.get_tensor_shape(input_name))
    if expected_input_shape != tuple(dummy_input.shape):
        raise ValueError(
            f"Engine expects {expected_input_shape}, dummy input is {tuple(dummy_input.shape)}"
        )
    device_input = dummy_input.to(
        dtype=torch_dtype(torch, trt, engine.get_tensor_dtype(input_name))
    ).contiguous()
    output_shape = tuple(int(v) for v in context.get_tensor_shape(output_name))
    device_output = torch.empty(
        output_shape,
        dtype=torch_dtype(torch, trt, engine.get_tensor_dtype(output_name)),
        device=device,
    ).contiguous()
    context.set_tensor_address(input_name, device_input.data_ptr())
    context.set_tensor_address(output_name, device_output.data_ptr())
    stream = torch.cuda.current_stream(device)
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT validation inference failed")
    stream.synchronize()

    native_model = models.make(checkpoint["model"], load_sd=True).to(device).eval()
    with torch.inference_mode():
        reference = native_model(dummy_input, dummy_scale).float()
    candidate = device_output.float()
    trt_diff = (candidate - reference).abs()
    report.update(
        {
            "tensorrt_max_abs_error": float(trt_diff.max().item()),
            "tensorrt_mean_abs_error": float(trt_diff.mean().item()),
            "tensorrt_allclose_atol_rtol_2e-2": bool(
                torch.allclose(candidate, reference, atol=2e-2, rtol=2e-2)
            ),
        }
    )
    write_report(report_path, report)
    if not report["tensorrt_allclose_atol_rtol_2e-2"]:
        raise RuntimeError(
            "TensorRT output failed the atol=rtol=2e-2 correctness check"
        )


if __name__ == "__main__":
    main()
