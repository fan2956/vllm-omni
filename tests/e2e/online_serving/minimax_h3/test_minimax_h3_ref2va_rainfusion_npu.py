# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real-video Ref2VA serving smoke with the A3 RainFusion benchmark configuration."""

import json
import os
import subprocess
from pathlib import Path

import av
import numpy as np
import pytest
import requests

from tests.dfx.conftest import _build_serve_args
from tests.helpers.assertions import assert_video_audio_valid, assert_video_valid
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer
from vllm_omni.transformers_utils.repo_utils import hf_api

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

TEST_NAME = "test_minimax_h3_ref2va_rainfusion_laser_4npu_a3"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "dfx/perf/tests/test_minimax_h3_vllm_omni.json"


@hardware_test(res={"npu": "A3"}, num_cards=4)
def test_ref2va_rainfusion_laser_video(tmp_path: Path) -> None:
    """Exercise the same real-weight server configuration used by the benchmark."""
    assert os.environ.get("MINDIE_SD_FA_TYPE") == "ascend_laser_attention"
    configs = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    matches = [config for config in configs if config["test_name"] == TEST_NAME]
    assert len(matches) == 1, f"Expected one benchmark configuration for {TEST_NAME}"
    config = matches[0]
    server_config = config["server_params"]
    workload = config["benchmark_params"][0]["extra_body"]

    model = os.environ.get("VLLM_TEST_MINIMAX_H3_REF2VA_MODEL")
    if not model:
        model = str(
            Path(
                hf_api().snapshot_download(
                    "MiniMaxAI/MiniMax-H3",
                    allow_patterns=["Ref2VA/**"],
                    local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
                )
            )
            / "Ref2VA"
        )

    artifact_dir = Path(os.environ.get("MINIMAX_H3_REF2VA_ARTIFACT_DIR", str(tmp_path)))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reference_path = artifact_dir / "reference.mp4"
    output_path = artifact_dir / "output.mp4"
    output_path.unlink(missing_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=832x480:rate=30:duration=5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(reference_path),
        ],
        check=True,
    )

    with OmniServer(model, _build_serve_args(server_config["serve_args"]), startup_timeout=2400) as server:
        data = {
            "model": server.model,
            "prompt": "A colorful moving scene with synchronized ambient sound.",
            **{
                key: str(workload[key])
                for key in ("width", "height", "fps", "num_inference_steps", "seed", "flow_shift")
            },
            "extra_params": workload["extra_params"],
        }
        with reference_path.open("rb") as reference:
            response = requests.post(
                f"http://{server.host}:{server.port}/v1/videos/sync",
                data=data,
                files=[("input_references", ("reference.mp4", reference, "video/mp4"))],
                headers={"Accept": "video/mp4"},
                timeout=3600,
            )
        response.raise_for_status()
        assert response.headers.get("content-type", "").startswith("video/mp4")
        assert response.content, "Ref2VA returned an empty MP4"
        output_path.write_bytes(response.content)

    metadata = assert_video_valid(output_path, width=workload["width"], height=workload["height"], fps=workload["fps"])
    assert metadata["num_frames"] == workload["num_frames"]
    assert_video_audio_valid(output_path, sample_rate=32000, channels=2)
    with av.open(str(output_path)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    assert frames == workload["num_frames"], f"Decoded {frames} video frames"
    with av.open(str(output_path)) as container:
        samples = 0
        for frame in container.decode(audio=0):
            samples += frame.samples
            assert np.isfinite(frame.to_ndarray()).all(), "Audio contains non-finite samples"
    assert samples > 0, "Ref2VA audio stream is empty"
