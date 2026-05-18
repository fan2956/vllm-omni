# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Progress bar mixin for diffusion pipelines.

Provides a diffusers-compatible progress_bar() method that wraps tqdm,
automatically disabling output on non-zero ranks in distributed settings.
"""

import time
from typing import Any, Callable

import torch
from tqdm.auto import tqdm


class ProgressBarMixin:
    """Mixin that provides a progress bar for denoising loops.

    Usage in pipeline:
        class MyPipeline(nn.Module, CFGParallelMixin, ProgressBarMixin):
            def diffuse(self, ...):
                with self.progress_bar(total=num_steps) as pbar:
                    for i, t in enumerate(timesteps):
                        ...
                        pbar.update()
    """

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        config = dict(self._progress_bar_config)
        # Only show progress bar on rank 0 in distributed settings
        if "disable" not in config:
            config["disable"] = not _is_rank_zero()

        if iterable is not None:
            return tqdm(iterable, **config)
        elif total is not None:
            return tqdm(total=total, **config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    def set_progress_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self._progress_callback = callback
        self._progress_started_at = None

    def report_step_progress(self, current_step: int, total_steps: int) -> None:
        callback = getattr(self, "_progress_callback", None)
        if callback is None or not _is_rank_zero():
            return
        total_steps = max(int(total_steps), 1)
        current_step = max(0, min(int(current_step), total_steps))
        now = time.perf_counter()
        started_at = getattr(self, "_progress_started_at", None)
        if started_at is None:
            started_at = now
            self._progress_started_at = started_at
        elapsed_s = max(0.0, now - started_at)
        seconds_per_step = elapsed_s / current_step if current_step > 0 else None
        callback(
            {
                "type": "progress",
                "current_step": current_step,
                "total_steps": total_steps,
                "percent": int((current_step / total_steps) * 100),
                "elapsed_s": elapsed_s,
                "seconds_per_step": seconds_per_step,
            }
        )


def _is_rank_zero() -> bool:
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0
