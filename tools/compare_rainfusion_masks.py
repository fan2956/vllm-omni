#!/usr/bin/env python3
"""Reconstruct CP RainFusion masks and compare them with full-Q trace masks."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch


def _load(directory: Path):
    traces = []
    for path in sorted(directory.glob("rainfusion_mask_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["path"] = path
        traces.append(payload)
    return traces


def _key(trace):
    return trace["step"], trace["layer"]


def _reconstruct_cp(traces):
    first = traces[0]
    block_size = first["block_size"]
    global_blocks = (first["kv_valid_len"] + block_size - 1) // block_size
    mask = torch.zeros(
        (*first["mask"].shape[:2], global_blocks, first["mask"].shape[-1]), dtype=torch.int8
    )
    filled = torch.zeros(global_blocks, dtype=torch.bool)
    for trace in traces:
        if trace["block_size"] != block_size or trace["kv_valid_len"] != first["kv_valid_len"]:
            raise ValueError("CP trace group has inconsistent block geometry")
        start = trace["q_global_start"] // block_size
        local_blocks = trace["mask"].shape[-2]
        end = start + local_blocks
        if end > global_blocks or filled[start:end].any():
            raise ValueError(f"invalid or overlapping CP rows in {trace['path']}")
        mask[:, :, start:end, :] = trace["mask"]
        filled[start:end] = True
    if not filled.all():
        missing = torch.where(~filled)[0].tolist()
        raise ValueError(f"CP trace group is missing global Q blocks: {missing}")
    return mask


def _reconstruct_full_q(traces):
    """Restore USP's head-sharded full-Q mask to the original head order."""
    ordered = sorted(traces, key=lambda trace: trace["rank"])
    first = ordered[0]
    expected_ranks = int(first["world_size"])
    if len(ordered) != expected_ranks:
        ranks = [trace["rank"] for trace in ordered]
        raise ValueError(
            f"full-Q trace is missing USP ranks: got {ranks}, expected {expected_ranks} ranks"
        )
    if [trace["rank"] for trace in ordered] != list(range(expected_ranks)):
        raise ValueError("full-Q trace ranks must be contiguous and start at zero")
    for trace in ordered[1:]:
        if (
            trace["block_size"] != first["block_size"]
            or trace["kv_valid_len"] != first["kv_valid_len"]
            or trace["mask"].shape[0] != first["mask"].shape[0]
            or trace["mask"].shape[2:] != first["mask"].shape[2:]
        ):
            raise ValueError("full-Q trace group has inconsistent block geometry")
    return torch.cat([trace["mask"] for trace in ordered], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-q-dir", type=Path, required=True, help="USP/full-Q trace directory")
    parser.add_argument("--cp-dir", type=Path, required=True, help="AllGather-KV CP trace directory")
    args = parser.parse_args()
    full_groups = defaultdict(list)
    cp_groups = defaultdict(list)
    for trace in _load(args.full_q_dir):
        full_groups[_key(trace)].append(trace)
    for trace in _load(args.cp_dir):
        cp_groups[_key(trace)].append(trace)
    common = sorted(full_groups.keys() & cp_groups.keys())
    if not common:
        raise ValueError("no common (step, layer) trace files")
    for key in common:
        reference = _reconstruct_full_q(full_groups[key])
        reconstructed = _reconstruct_cp(cp_groups[key])
        if reference.shape != reconstructed.shape:
            raise ValueError(f"step={key[0]} layer={key[1]} shape mismatch: {reference.shape} vs {reconstructed.shape}")
        diff = reference != reconstructed
        changed_rows = torch.where(diff.any(dim=(0, 1, 3)))[0].tolist()
        print(
            f"step={key[0]} layer={key[1]} diff={int(diff.sum())}/{diff.numel()} "
            f"ratio={diff.float().mean().item():.8f} changed_q_blocks={changed_rows}"
        )


if __name__ == "__main__":
    main()
