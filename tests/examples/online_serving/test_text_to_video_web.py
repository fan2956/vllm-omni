# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from examples.online_serving.text_to_video_web.app import DEFAULT_NEGATIVE_PROMPT, CreateVideoRequest, create_app

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeOmniClient:
    def __init__(self, server_url: str = "http://fake-omni") -> None:
        self.server_url = server_url
        self.created_forms: list[dict[str, str]] = []
        self.health_calls = 0

    async def check_health(self):
        self.health_calls += 1
        return {"ok": True, "server_url": self.server_url}

    async def create_video(self, request: CreateVideoRequest):
        self.created_forms.append(request.to_omni_form())
        return {"id": "video_gen_test", "status": "queued", "object": "video"}

    async def get_video(self, video_id: str):
        return {
            "id": video_id,
            "status": "failed",
            "progress": 0,
            "step_progress": {
                "current_step": 3,
                "total_steps": 40,
                "percent": 8,
                "seconds_per_step": 1.2,
            },
            "error": {"code": "RuntimeError", "message": "boom"},
        }


def _test_client(
    fake_client: FakeOmniClient | None = None,
    compare_client: FakeOmniClient | None = None,
) -> TestClient:
    app = create_app("http://fake-omni", "http://fake-compare")
    if fake_client is not None:
        app.state.omni_clients = {
            "default": fake_client,
            "compare": compare_client or FakeOmniClient("http://fake-compare"),
        }
    return TestClient(app)


def test_create_video_request_defaults_are_forwarded_as_omni_form():
    request = CreateVideoRequest(prompt="  a bright racing car  ")

    assert request.to_omni_form() == {
        "prompt": "a bright racing car",
        "size": "832x480",
        "fps": "12",
        "num_frames": "61",
        "guidance_scale": "1.0",
        "flow_shift": "5.0",
        "num_inference_steps": "40",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "enable_frame_interpolation": "true",
        "frame_interpolation_model_path": "/home/zf/vllm-omni/elfgum",
        "frame_interpolation_exp": "1",
        "frame_interpolation_scale": "1.0",
    }


def test_create_video_request_accepts_exact_size_choices():
    request = CreateVideoRequest(prompt="wide shot", size="1280x720")

    assert request.size == "1280x720"
    assert request.to_omni_form()["size"] == "1280x720"


def test_create_video_request_rejects_asterisk_size_without_normalizing():
    with pytest.raises(ValueError, match="size must be one of"):
        CreateVideoRequest(prompt="wide shot", size="1280*720")


def test_create_video_request_omits_seed_when_missing_or_negative_one():
    assert "seed" not in CreateVideoRequest(prompt="random seed").to_omni_form()
    assert "seed" not in CreateVideoRequest(prompt="random seed", seed=-1).to_omni_form()
    assert CreateVideoRequest(prompt="fixed seed", seed=42).to_omni_form()["seed"] == "42"


def test_create_video_request_accepts_default_and_compare_server_ids():
    assert CreateVideoRequest(prompt="primary", server_id="default").server_id == "default"
    assert CreateVideoRequest(prompt="secondary", server_id="compare").server_id == "compare"
    with pytest.raises(ValueError, match="server_id must be one of"):
        CreateVideoRequest(prompt="bad", server_id="other")


def test_disabled_frame_interpolation_only_forwards_enable_flag():
    request = CreateVideoRequest(prompt="no interpolation", enable_frame_interpolation=False)

    form = request.to_omni_form()
    assert form["enable_frame_interpolation"] == "false"
    assert "frame_interpolation_model_path" not in form
    assert "frame_interpolation_exp" not in form
    assert "frame_interpolation_scale" not in form


def test_create_video_proxies_to_default_omni_client():
    fake_client = FakeOmniClient()
    with _test_client(fake_client) as client:
        response = client.post("/api/videos", json={"prompt": "a lighthouse", "fps": 16, "num_inference_steps": 50})

    assert response.status_code == 200
    assert response.json()["id"] == "video_gen_test"
    assert "step_progress" not in response.json()
    assert fake_client.created_forms[0]["prompt"] == "a lighthouse"
    assert fake_client.created_forms[0]["fps"] == "16"
    assert fake_client.created_forms[0]["flow_shift"] == "5.0"


def test_create_video_proxies_to_compare_omni_client():
    default_client = FakeOmniClient("http://fake-omni")
    compare_client = FakeOmniClient("http://fake-compare")
    with _test_client(default_client, compare_client) as client:
        response = client.post("/api/videos", json={"server_id": "compare", "prompt": "a lighthouse"})

    assert response.status_code == 200
    assert default_client.created_forms == []
    assert compare_client.created_forms[0]["prompt"] == "a lighthouse"


def test_status_polling_passes_failed_job_payload_through():
    with _test_client(FakeOmniClient()) as client:
        response = client.get("/api/videos/video_gen_test")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["step_progress"]["current_step"] == 3
    assert response.json()["step_progress"]["seconds_per_step"] == 1.2
    assert response.json()["error"]["message"] == "boom"


def test_status_polling_routes_by_server_id():
    default_client = FakeOmniClient("http://fake-omni")
    compare_client = FakeOmniClient("http://fake-compare")
    with _test_client(default_client, compare_client) as client:
        response = client.get("/api/videos/video_gen_test", params={"server_id": "compare"})

    assert response.status_code == 200
    assert response.json()["status"] == "failed"


def test_content_endpoint_streams_from_omni_server(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []

    class FakeUpstreamResponse:
        status_code = 200
        headers = {"content-length": "7"}

        async def aiter_bytes(self):
            events.append("stream")
            yield b"mp4"
            yield b"data"

        async def aclose(self):
            events.append("response_closed")

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            events.append(f"client:{kwargs.get('timeout')}")

        def build_request(self, method: str, url: str, **kwargs):
            events.append(f"{method} {url}")
            headers = kwargs.get("headers", {})
            events.append(f"range:{headers.get('range')}")
            return SimpleNamespace(method=method, url=url, headers=headers)

        async def send(self, request, stream: bool = False):
            events.append(f"send:{stream}")
            return FakeUpstreamResponse()

        async def aclose(self):
            events.append("client_closed")

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    with _test_client() as client:
        response = client.get(
            "/api/videos/video_gen_test/content",
            params={"server_id": "compare"},
            headers={"range": "bytes=0-6"},
        )

    assert response.status_code == 200
    assert response.content == b"mp4data"
    assert "GET http://fake-compare/v1/videos/video_gen_test/content" in events
    assert "range:bytes=0-6" in events
    assert "send:True" in events
    assert "response_closed" in events
    assert "client_closed" in events


def test_health_reports_omni_status():
    default_client = FakeOmniClient()
    compare_client = FakeOmniClient("http://fake-compare")
    with _test_client(default_client, compare_client) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["web"]["ok"] is True
    assert response.json()["omni"]["ok"] is True
    assert response.json()["servers"]["default"]["title"] == "vLLM-Omni (Speed By MindIE SD)"
    assert response.json()["servers"]["compare"]["title"] == "vLLM-Omni"
    assert default_client.health_calls == 1
    assert compare_client.health_calls == 1


def test_health_can_skip_compare_server():
    default_client = FakeOmniClient()
    compare_client = FakeOmniClient("http://fake-compare")
    with _test_client(default_client, compare_client) as client:
        response = client.get("/api/health", params={"include_compare": "false"})

    assert response.status_code == 200
    assert set(response.json()["servers"]) == {"default"}
    assert default_client.health_calls == 1
    assert compare_client.health_calls == 0
