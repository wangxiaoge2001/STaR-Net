#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

overlap=64
resolution_z=64

# Defaults can be overridden with command-line options.
backend="pytorch"
trt_cuda_graph=0
gpu="0"
config_suffix="train_starnet"
epoch_num=2000
seq_len=4
inp_size=450
data_path="$REPO_ROOT/demo_data"
begin_frame=1
end_frame=12

usage() {
  cat <<'EOF'
Usage: bash_test.sh [OPTIONS]

  --backend NAME          Inference backend: pytorch or tensorrt (default: pytorch)
  --trt-cuda-graph        Enable fixed-address CUDA Graphs for TensorRT
  --gpu ID                GPU ID (default: 0)
  --config-suffix NAME    Checkpoint directory under save/ (default: train_starnet)
  --epoch-num N           Checkpoint epoch (default: 2000)
  --seq-len N             Sequence length (default: 4)
  --inp-size N            Input patch size (default: 450)
  --data-path PATH        Test data root (default: ../demo_data/)
  --begin-frame N         First frame, inclusive (default: 1)
  --end-frame N           Last frame, inclusive (default: 12)

  -h, --help              Show this help message

Example:
  bash bash_test.sh --gpu 0 --epoch-num 2000 --data-path ../demo_data/ --backend pytorch
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      backend="$2"
      shift 2
      ;;
    --trt-cuda-graph|--trt_cuda_graph)
      trt_cuda_graph=1
      shift
      ;;
    --gpu)
      gpu="$2"
      shift 2
      ;;
    --config-suffix)
      config_suffix="$2"
      shift 2
      ;;
    --epoch-num|--epoch_num)
      epoch_num="$2"
      shift 2
      ;;
    --seq-len|--seq_len)
      seq_len="$2"
      shift 2
      ;;
    --inp-size|--inp_size)
      inp_size="$2"
      shift 2
      ;;
    --data-path|--data_path)
      data_path="$2"
      shift 2
      ;;
    --begin-frame|--begin_frame)
      begin_frame="$2"
      shift 2
      ;;
    --end-frame|--end_frame)
      end_frame="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

input_str="input"

save_individual=0
save_tif=1
save_MIPz=1
test_scrip="$SCRIPT_DIR/test.py"

echo "Starting inference..."

save_file_name=frame
usingreplacestr="$config_suffix"-epoch"$epoch_num"/
usingcodefolder="$SCRIPT_DIR/"
usingmodel="$REPO_ROOT/save/$config_suffix/epoch-$epoch_num.pth"
trt_engine_dir="$REPO_ROOT/save/$config_suffix/tensorrt_models_epoch$epoch_num"

save_prefix="test/"
usingmodel_full=$(realpath "$usingmodel")
usingcodefolder_full=$(realpath "$usingcodefolder")
data_path_full=$(realpath "$data_path")

echo "=========================================="
echo "Inference configuration:"
echo "  usingmodel:       $usingmodel_full"
echo "  usingcodefolder:  $usingcodefolder_full"
echo "  config_suffix:    $config_suffix"
echo "  epoch_num:        $epoch_num"
echo "  seq_len:          $seq_len"
echo "  inp_size:         $inp_size"
echo "  gpu:              $gpu"
echo "  backend:          $backend"
echo "  trt_cuda_graph:   $trt_cuda_graph"
echo "  data_path:        $data_path_full/$input_str"
echo "=========================================="

trt_cuda_graph_args=()
if [[ "$trt_cuda_graph" -eq 1 ]]; then
  trt_cuda_graph_args+=(--trt_cuda_graph)
fi

python "$test_scrip" --gpu "$gpu" --datapath "$data_path_full/$input_str" --savefolder "$data_path_full/$save_prefix${usingreplacestr}" --model "$usingmodel" --resolution_z "$resolution_z" --inp_size "$inp_size" --overlap "$overlap" --startframe 0 --codefolder "$usingcodefolder" --replacestr "$usingreplacestr" --save_file_name "$save_file_name" --begin_frame "$begin_frame" --end_frame "$end_frame" --save_tif "$save_tif" --save_MIPz "$save_MIPz" --save_individual "$save_individual" --seq_len "$seq_len" --backend "$backend" --trt_engine_dir "$trt_engine_dir" "${trt_cuda_graph_args[@]}"

programName=${0##*/}
cp "$SCRIPT_DIR/$programName" "$data_path_full/$save_prefix${usingreplacestr}"
cp "$test_scrip" "$data_path_full/$save_prefix${usingreplacestr}"


echo "Inference completed."
