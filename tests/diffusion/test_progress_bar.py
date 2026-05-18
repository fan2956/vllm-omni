# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def test_progress_bar_mixin_reports_step_progress():
    pipeline = ProgressBarMixin()
    events = []

    pipeline.set_progress_callback(events.append)
    pipeline.report_step_progress(1, 4)
    pipeline.report_step_progress(4, 4)

    assert len(events) == 2
    assert events[0]["type"] == "progress"
    assert events[0]["current_step"] == 1
    assert events[0]["total_steps"] == 4
    assert events[0]["percent"] == 25
    assert events[0]["elapsed_s"] >= 0
    assert events[0]["seconds_per_step"] >= 0
    assert events[1]["current_step"] == 4
    assert events[1]["total_steps"] == 4
    assert events[1]["percent"] == 100
    assert events[1]["seconds_per_step"] >= 0
