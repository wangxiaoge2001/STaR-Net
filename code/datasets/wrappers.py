import random

import numpy as np
import torch

from datasets import register


class _SequenceWrapperBase:
    """Preprocess paired light-field and volume sequences for STaR-Net."""

    def __init__(
        self,
        dataset,
        randomSeed=None,
        inp_size=None,
        RPN_noise=None,
        RGN_noise=None,
        balance_percent=0.1,
        max_percentile=100,
        occlusion_probability=None,
        seq_len=4,
        rand_reverse=False,
        normalize_65535=False,
        **_unused,
    ):
        self.dataset = dataset
        self.inp_size = inp_size
        self.RPN_noise = RPN_noise
        self.RGN_noise = RGN_noise
        self.balance_percent = balance_percent
        self.max_percentile = max_percentile
        self.occlusion_probability = occlusion_probability
        self.seq_len = seq_len
        self.rand_reverse = rand_reverse
        self.normalize_65535 = normalize_65535

        if randomSeed is not None:
            torch.manual_seed(randomSeed)

    def get_max_value_dataset_1(self):
        return self.dataset.get_max_value_dataset_1()

    def get_max_value_dataset_2(self):
        return self.dataset.get_max_value_dataset_2()

    @staticmethod
    def _ensure_tensor_sequence(data):
        if torch.is_tensor(data):
            return data
        if isinstance(data, np.ndarray):
            return torch.as_tensor(data)
        if isinstance(data, (list, tuple)):
            if not data:
                raise ValueError("Expected a non-empty sequence")
            return torch.stack([torch.as_tensor(item) for item in data], dim=0)
        raise TypeError(f"Unsupported sequence type: {type(data)!r}")

    def _maybe_reverse_sequence(self, lfstack, volume):
        if self.rand_reverse and random.randint(0, 2) == 0:
            return torch.flip(lfstack, dims=[0]), torch.flip(volume, dims=[0])
        return lfstack, volume

    @staticmethod
    def _sanitize_pair(lfstack, volume):
        return (
            torch.nan_to_num(lfstack.clone(), nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(volume.clone(), nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _normalization_scales(self, lfstack, volume):
        if self.normalize_65535:
            return 65535.0, 65535.0
        return (
            float(np.percentile(lfstack, self.max_percentile)),
            float(np.percentile(volume, self.max_percentile)),
        )

    def _crop_balanced_patch(self, lfstack, volume, scale):
        if self.inp_size is None:
            return lfstack, volume

        crop_h = crop_w = self.inp_size
        input_h, input_w = lfstack.shape[-2:]
        if crop_h > input_h or crop_w > input_w:
            raise ValueError(
                f"inp_size={self.inp_size} exceeds input size {input_h}x{input_w}"
            )

        target_h = round(crop_h * scale)
        target_w = round(crop_w * scale)
        volume_mean = volume.mean()

        def crop_once():
            h0 = random.randint(0, input_h - crop_h)
            w0 = random.randint(0, input_w - crop_w)
            H0, W0 = round(h0 * scale), round(w0 * scale)
            return (
                lfstack[:, :, h0:h0 + crop_h, w0:w0 + crop_w],
                volume[:, :, H0:H0 + target_h, W0:W0 + target_w],
            )

        cropped_lfstack, cropped_volume = crop_once()
        attempts = 0
        while (
            cropped_volume.mean() < volume_mean * self.balance_percent
            and attempts < 100
            and random.randint(0, 20) != 0
        ):
            cropped_lfstack, cropped_volume = crop_once()
            attempts += 1
        return cropped_lfstack, cropped_volume

    def _apply_noise(self, lfstack, lfstack_max):
        scale = max(float(lfstack_max), 1e-6)
        if self.RPN_noise is not None:
            choices = torch.as_tensor(self.RPN_noise, dtype=torch.float32)
            poisson_lambda = choices[torch.multinomial(torch.ones_like(choices), 1)].item()
            lfstack = torch.poisson(lfstack.clamp_min(0) / scale * poisson_lambda)
            lfstack = lfstack * scale / poisson_lambda

        if self.RGN_noise is not None:
            lfstack = lfstack / scale * 65536
            gaussian_mean = np.random.randint(0, self.RGN_noise[0])
            gaussian_sigma = np.random.randint(
                self.RGN_noise[0] // 2, self.RGN_noise[0]
            ) ** 0.5
            lfstack = (
                lfstack
                + gaussian_mean
                + torch.randn_like(lfstack) * gaussian_sigma
            )
            lfstack = lfstack.clamp_min(0) * scale / 65536
        return lfstack

    @staticmethod
    def _generate_occlusion_matrix(size):
        row = np.ones(3 * size)
        col = np.ones(3 * size)

        row_min = np.random.randint(5, 2 * size)
        row_max = np.random.randint(row_min + 1, 3 * size - 5)
        col_min = np.random.randint(5, 2 * size)
        col_max = np.random.randint(col_min + 1, 3 * size - 5)

        outer_row_min = np.random.randint(0, row_min - 2)
        outer_row_max = np.random.randint(row_max + 2, 3 * size)
        outer_col_min = np.random.randint(0, col_min - 2)
        outer_col_max = np.random.randint(col_max + 2, 3 * size)

        row[row_min:row_max] = 0
        row[row_max:outer_row_max] = np.linspace(0, 1, outer_row_max - row_max)
        row[outer_row_min:row_min] = np.linspace(1, 0, row_min - outer_row_min)
        col[col_min:col_max] = 0
        col[col_max:outer_col_max] = np.linspace(0, 1, outer_col_max - col_max)
        col[outer_col_min:col_min] = np.linspace(1, 0, col_min - outer_col_min)

        matrix = np.maximum.outer(row, col)
        return torch.from_numpy(matrix[size:2 * size, size:2 * size])

    def _apply_occlusion(self, lfstack):
        if self.occlusion_probability is None:
            return lfstack

        occlusion_count = int(
            torch.multinomial(
                torch.as_tensor(self.occlusion_probability, dtype=torch.float32),
                1,
            ).item()
        )
        view_count = lfstack.shape[1]
        obscured_views = set()
        for _ in range(min(occlusion_count, view_count)):
            obscured_view = np.random.randint(0, view_count)
            while obscured_view in obscured_views:
                obscured_view = np.random.randint(0, view_count)
            obscured_views.add(obscured_view)
            mask = self._generate_occlusion_matrix(lfstack.shape[-1]).to(
                device=lfstack.device,
                dtype=lfstack.dtype,
            )
            lfstack[:, obscured_view] *= mask
        return lfstack

    def _build_sample(self, lfstack, volume):
        lfstack = self._ensure_tensor_sequence(lfstack)
        volume = self._ensure_tensor_sequence(volume)
        lfstack, volume = self._maybe_reverse_sequence(lfstack, volume)
        lfstack, volume = self._sanitize_pair(lfstack, volume)
        lfstack_max, volume_max = self._normalization_scales(lfstack, volume)
        scale = volume.shape[-1] / lfstack.shape[-1]
        lfstack, volume = self._crop_balanced_patch(lfstack, volume, scale)
        lfstack = self._apply_noise(lfstack, lfstack_max)
        lfstack = (lfstack / max(lfstack_max, 1e-6)).clamp(0, 1)
        volume = (volume / max(volume_max, 1e-6)).clamp(0, 1)
        lfstack = self._apply_occlusion(lfstack)
        return {
            "inp": lfstack,
            "scale": torch.tensor([scale], dtype=torch.float32),
            "gt": volume,
        }

    def __len__(self):
        return len(self.dataset)


@register("rlfm-tpm-trans-occlusion-seq3-random-sample")
class STaRNetRandomSequenceWrapper(_SequenceWrapperBase):
    def __init__(self, sample_factor=4, **kwargs):
        super().__init__(**kwargs)
        if sample_factor < 1:
            raise ValueError("sample_factor must be at least 1")
        self.sample_factor = sample_factor

    def __getitem__(self, idx):
        first = idx * self.sample_factor
        source_length = len(getattr(self.dataset, "dataset_1", self.dataset))
        last = min((idx + 1) * self.sample_factor, source_length) - 1
        true_idx = random.randint(first, last)
        lfstack, volume = self.dataset[true_idx]
        return self._build_sample(lfstack, volume)


@register("rlfm-tpm-trans-occlusion-seq3")
class STaRNetSequenceWrapper(_SequenceWrapperBase):
    def __getitem__(self, idx):
        lfstack, volume = self.dataset[idx]
        return self._build_sample(lfstack, volume)
