import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from tifffile import imread


# User-adjustable parameters.
src_dir_starnet = "../demo_data/test/train_starnet-epoch2000/"
src_dir_rl = "../demo_data/test/RL/"

# Frame IDs to compare, for example [1, 4, 8].
frame_ids = [1]

# Use None to estimate a display limit from each method's volume percentiles.
vmin_starnet = 0
vmax_starnet = 10000
vmin_rl = 0
vmax_rl = 2000
auto_percentiles = (0.5, 99.8)

# Z slices shown beside the projections; use None for the center slice.
slice_z_list = [35, 60]

# Number of slices excluded from each end before the MIPz projection.
mipz_cut_edge = 5

output_dir = "./visualization_results"
save_dpi = 220
lut_csv_path = r"./LUT_green_fire_blue.csv"


def load_lut_colormap(lut_path: str | Path) -> LinearSegmentedColormap:
    lut_path = Path(lut_path)
    if not lut_path.exists():
        raise FileNotFoundError(f"Cannot find LUT csv: {lut_path}")

    lut = np.loadtxt(str(lut_path), delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype=np.float32)
    if lut.ndim != 2 or lut.shape[1] != 3:
        raise ValueError(f"Expected LUT csv columns Red,Green,Blue, but got shape {lut.shape}: {lut_path}")

    lut = np.clip(lut, 0, 255) / 255.0
    return LinearSegmentedColormap.from_list("green_fire_blue", lut, N=lut.shape[0])


def read_volume(tif_path: Path) -> np.ndarray:
    if not tif_path.exists():
        raise FileNotFoundError(f"Cannot find tif file: {tif_path}")

    volume = imread(str(tif_path))
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D tif, but got shape {volume.shape}: {tif_path}")
    return volume


def normalize_index(idx: int | None, size: int) -> int:
    if idx is None:
        return size // 2
    if idx < 0:
        idx += size
    return int(np.clip(idx, 0, size - 1))


def normalize_z_slice_list(z_indices: list[int | None] | tuple[int | None, ...] | None, depth: int) -> list[int]:
    if z_indices is None:
        z_indices = [None]
    return [normalize_index(z_idx, depth) for z_idx in z_indices]


def get_mipz_volume(volume: np.ndarray, cut_edge: int) -> np.ndarray:
    cut_edge = max(int(cut_edge), 0)
    if cut_edge == 0 or cut_edge * 2 >= volume.shape[0]:
        return volume
    return volume[cut_edge:-cut_edge, :, :]


def get_views(volume: np.ndarray, z_indices: list[int | None] | tuple[int | None, ...] | None) -> dict[str, np.ndarray]:
    views = {
        "MIPz": np.max(get_mipz_volume(volume, mipz_cut_edge), axis=0),
        "MIPy": np.max(volume, axis=1),
    }
    for z_idx in normalize_z_slice_list(z_indices, volume.shape[0]):
        views[f"slice z={z_idx}"] = volume[z_idx, :, :]
    return views


def robust_display_range(volumes: list[np.ndarray], user_vmin: float | None, user_vmax: float | None) -> tuple[float, float]:
    if user_vmin is not None and user_vmax is not None:
        return float(user_vmin), float(user_vmax)

    pixels = np.concatenate([v[np.isfinite(v)].ravel() for v in volumes])
    if pixels.size == 0:
        raise ValueError("No finite pixel values found.")

    p_low, p_high = np.percentile(pixels, auto_percentiles)
    low = p_low if user_vmin is None else user_vmin
    high = p_high if user_vmax is None else user_vmax
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def add_image(ax, image: np.ndarray, title: str, cmap, min_value: float, max_value: float) -> None:
    ax.imshow(
        image,
        cmap=cmap,
        vmin=min_value,
        vmax=max_value,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_one_frame(
    starnet_volume: np.ndarray,
    rl_volume: np.ndarray,
    frame_id: int,
    save_path: Path,
    cmap,
    starnet_min: float,
    starnet_max: float,
    rl_min: float,
    rl_max: float,
) -> None:
    starnet_views = get_views(starnet_volume, slice_z_list)
    rl_views = get_views(rl_volume, slice_z_list)

    total_cols = len(starnet_views)
    fig = plt.figure(figsize=(3.1 * total_cols, 6.2))
    gs = fig.add_gridspec(2, total_cols, left=0.04, right=0.99, bottom=0.06, top=0.88, wspace=0.18, hspace=0.28)
    fig.suptitle(f"frame_{frame_id}.tif", fontsize=12, y=0.99)

    for col_offset, (view_name, image) in enumerate(starnet_views.items()):
        ax = fig.add_subplot(gs[0, col_offset])
        add_image(ax, image, f"STaR-Net {view_name}", cmap, starnet_min, starnet_max)

    for col_offset, (view_name, image) in enumerate(rl_views.items()):
        ax = fig.add_subplot(gs[1, col_offset])
        add_image(ax, image, f"RL {view_name}", cmap, rl_min, rl_max)

    fig.add_artist(
        plt.Line2D(
            [0.04, 0.99],
            [0.47, 0.47],
            transform=fig.transFigure,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
    )

    fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize and compare 3D tif volumes from STaR-Net and RL.")
    parser.add_argument("--src_dir_starnet", default=src_dir_starnet, help="Directory containing STaR-Net frame_*.tif files.")
    parser.add_argument("--src_dir_rl", default=src_dir_rl, help="Directory containing RL frame_*.tif files.")
    parser.add_argument("--frame_ids", nargs="+", type=int, default=frame_ids, help="Frame ids to visualize, e.g. 1 4 8.")
    parser.add_argument("--output_dir", default=output_dir, help="Directory to save visualization PNG files.")
    parser.add_argument("--lut_csv_path", default=lut_csv_path, help="CSV LUT with columns Index,Red,Green,Blue.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    starnet_dir = Path(args.src_dir_starnet.strip())
    rl_dir = Path(args.src_dir_rl.strip())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = load_lut_colormap(args.lut_csv_path)
    for frame_id in args.frame_ids:
        starnet_path = starnet_dir / f"frame_{frame_id}.tif"
        rl_path = rl_dir / f"frame_{frame_id}.tif"

        starnet_volume = read_volume(starnet_path)
        rl_volume = read_volume(rl_path)
        if starnet_volume.shape != rl_volume.shape:
            print(f"Warning: frame_{frame_id} shape mismatch: STaR-Net {starnet_volume.shape}, RL {rl_volume.shape}")

        starnet_min, starnet_max = robust_display_range([starnet_volume], vmin_starnet, vmax_starnet)
        rl_min, rl_max = robust_display_range([rl_volume], vmin_rl, vmax_rl)
        save_path = out_dir / f"compare_frame_{frame_id}.png"
        visualize_one_frame(
            starnet_volume,
            rl_volume,
            frame_id,
            save_path,
            cmap,
            starnet_min,
            starnet_max,
            rl_min,
            rl_max,
        )
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
