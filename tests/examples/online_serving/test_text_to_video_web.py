# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from examples.online_serving.text_to_video_web.app import CreateVideoRequest, create_app

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeOmniClient:
    server_url = "http://fake-omni"

    def __init__(self) -> None:
        self.created_forms: list[dict[str, str]] = []

    async def check_health(self):
        return {"ok": True, "server_url": self.server_url}

    async def create_video(self, request: CreateVideoRequest):
        self.created_forms.append(request.to_omni_form())
        return {"id": "video_gen_test", "status": "queued", "object": "video"}

    async def get_video(self, video_id: str):
        return {
            "id": video_id,
            "status": "failed",
            "progress": 0,
            "error": {"code": "RuntimeError", "message": "boom"},
        }


def _test_client(fake_client: FakeOmniClient | None = None) -> TestClient:
    app = create_app("http://fake-omni")
    if fake_client is not None:
        app.state.omni_clients = {"default": fake_client}
    return TestClient(app)


def test_create_video_request_defaults_are_forwarded_as_omni_form():
    request = CreateVideoRequest(prompt="  a bright racing car  ")

    assert request.to_omni_form() == {
        "prompt": "a bright racing car",
        "size": "720x1280",
        "fps": "12",
        "num_frames": "61",
        "guidance_scale": "1.0",
        "flow_shift": "5.0",
        "num_inference_steps": "40",
        "seed": "42",
        "enable_frame_interpolation": "true",
        "frame_interpolation_model_path": "/home/zf/vllm-omni/elfgum",
        "frame_interpolation_exp": "1",
        "frame_interpolation_scale": "1.0",
    }


def test_create_video_proxies_to_default_omni_client():
    fake_client = FakeOmniClient()
    with _test_client(fake_client) as client:
        response = client.post("/api/videos", json={"prompt": "a lighthouse", "fps": 16})

    assert response.status_code == 200
    assert response.json()["id"] == "video_gen_test"
    assert fake_client.created_forms[0]["prompt"] == "a lighthouse"
    assert fake_client.created_forms[0]["fps"] == "16"


def test_status_polling_passes_failed_job_payload_through():
    with _test_client(FakeOmniClient()) as client:
        response = client.get("/api/videos/video_gen_test")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error"]["message"] == "boom"


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
        response = client.get("/api/videos/video_gen_test/content", headers={"range": "bytes=0-6"})

    assert response.status_code == 200
    assert response.content == b"mp4data"
    assert "GET http://fake-omni/v1/videos/video_gen_test/content" in events
    assert "range:bytes=0-6" in events
    assert "send:True" in events
    assert "response_closed" in events
    assert "client_closed" in events


def test_health_reports_omni_status():
    with _test_client(FakeOmniClient()) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["web"]["ok"] is True
    assert response.json()["omni"]["ok"] is True
