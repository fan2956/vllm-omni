# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small web UI for Wan2.2 text-to-video serving.

Run this inside the same container as the vLLM-Omni server and open the mapped
web port from the host browser.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

DEFAULT_OMNI_SERVER_URL = "http://127.0.0.1:8099"
STATIC_DIR = Path(__file__).with_name("static")
ALLOWED_VIDEO_SIZES = {"832x480", "1280x720"}
DEFAULT_FLOW_SHIFT = 5.0
DEFAULT_FRAME_INTERPOLATION_MODEL_PATH = "/home/zf/vllm-omni/elfgum"
DEFAULT_FRAME_INTERPOLATION_EXP = 1
DEFAULT_FRAME_INTERPOLATION_SCALE = 1.0


class CreateVideoRequest(BaseModel):
    """Browser-facing request shape.

    ``server_id`` is fixed to ``default`` for now. Keeping it in the API makes
    the future two-server comparison mode additive instead of a breaking change.
    """

    server_id: str = "default"
    prompt: str
    size: str = "832x480"
    fps: int = Field(default=12, ge=1)
    num_frames: int = Field(default=61, ge=1)
    guidance_scale: float = Field(default=1.0, ge=0.0)
    num_inference_steps: int = Field(default=40, ge=1)
    seed: int | None = 42
    negative_prompt: str | None = None
    enable_frame_interpolation: bool = True

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt is required")
        return value

    @field_validator("server_id")
    @classmethod
    def validate_server_id(cls, value: str) -> str:
        if value != "default":
            raise ValueError("Only server_id='default' is configured in this demo.")
        return value

    @field_validator("size")
    @classmethod
    def validate_size(cls, value: str) -> str:
        value = value.strip()
        if value not in ALLOWED_VIDEO_SIZES:
            supported = ", ".join(sorted(ALLOWED_VIDEO_SIZES))
            raise ValueError(f"size must be one of: {supported}")
        return value

    def to_omni_form(self) -> dict[str, str]:
        data: dict[str, Any] = self.model_dump(exclude={"server_id"})
        form: dict[str, str] = {}
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue
            if isinstance(value, bool):
                form[key] = "true" if value else "false"
            else:
                form[key] = str(value)
        form["flow_shift"] = str(DEFAULT_FLOW_SHIFT)
        if self.enable_frame_interpolation:
            form["frame_interpolation_model_path"] = DEFAULT_FRAME_INTERPOLATION_MODEL_PATH
            form["frame_interpolation_exp"] = str(DEFAULT_FRAME_INTERPOLATION_EXP)
            form["frame_interpolation_scale"] = str(DEFAULT_FRAME_INTERPOLATION_SCALE)
        return form


class OmniClient:
    """Thin client for a single vLLM-Omni OpenAI-compatible server."""

    def __init__(self, server_url: str, *, timeout_s: float = 60.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s

    async def create_video(self, request: CreateVideoRequest) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(f"{self.server_url}/v1/videos", data=request.to_omni_form())
        return _json_or_raise(response)

    async def get_video(self, video_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(f"{self.server_url}/v1/videos/{video_id}")
        return _json_or_raise(response)

    async def check_health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/v1/videos", params={"limit": 0})
            if response.status_code >= 400:
                return {"ok": False, "status_code": response.status_code, "detail": _response_detail(response)}
            return {"ok": True, "status_code": response.status_code, "server_url": self.server_url}
        except httpx.RequestError as exc:
            return {"ok": False, "server_url": self.server_url, "detail": str(exc)}


def _response_detail(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def _json_or_raise(response: httpx.Response) -> dict[str, Any]:
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=_response_detail(response))
    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Omni server returned a non-JSON response.") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Omni server returned an unexpected response shape.")
    return data


def _client_from_request(request: Request) -> OmniClient:
    return request.app.state.omni_clients["default"]


def create_app(omni_server_url: str | None = None) -> FastAPI:
    omni_server_url = omni_server_url or os.getenv("OMNI_SERVER_URL", DEFAULT_OMNI_SERVER_URL)

    app = FastAPI(title="Wan2.2 Text-to-Video Web Demo")
    app.state.omni_clients = {"default": OmniClient(omni_server_url)}

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        client = _client_from_request(request)
        return {"web": {"ok": True}, "omni": await client.check_health()}

    @app.post("/api/videos")
    async def create_video(request_data: CreateVideoRequest, request: Request) -> dict[str, Any]:
        client = _client_from_request(request)
        try:
            response = await client.create_video(request_data)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to connect to omni server: {exc}") from exc
        return response

    @app.get("/api/videos/{video_id}")
    async def get_video(video_id: str, request: Request) -> dict[str, Any]:
        client = _client_from_request(request)
        try:
            response = await client.get_video(video_id)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to connect to omni server: {exc}") from exc
        return response

    @app.get("/api/videos/{video_id}/content")
    async def video_content(video_id: str, request: Request) -> Response:
        client = _client_from_request(request)
        upstream_client = httpx.AsyncClient(timeout=None)
        try:
            upstream_headers = {}
            range_header = request.headers.get("range")
            if range_header:
                upstream_headers["range"] = range_header
            upstream_request = upstream_client.build_request(
                "GET",
                f"{client.server_url}/v1/videos/{video_id}/content",
                headers=upstream_headers,
            )
            upstream_response = await upstream_client.send(upstream_request, stream=True)
        except httpx.RequestError as exc:
            await upstream_client.aclose()
            raise HTTPException(status_code=502, detail=f"Failed to connect to omni server: {exc}") from exc

        if upstream_response.status_code >= 400:
            content = await upstream_response.aread()
            content_type = upstream_response.headers.get("content-type", "text/plain")
            await upstream_response.aclose()
            await upstream_client.aclose()
            return Response(content=content, status_code=upstream_response.status_code, media_type=content_type)

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream_response.aiter_bytes():
                    yield chunk
            finally:
                await upstream_response.aclose()
                await upstream_client.aclose()

        headers = {}
        for header_name in ("content-length", "content-range", "accept-ranges"):
            header_value = upstream_response.headers.get(header_name)
            if header_value:
                headers[header_name] = header_value
        media_type = upstream_response.headers.get("content-type", "video/mp4")
        return StreamingResponse(
            relay(),
            status_code=upstream_response.status_code,
            media_type=media_type,
            headers=headers,
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wan2.2 text-to-video web demo")
    parser.add_argument("--host", default="0.0.0.0", help="Host for the web server")
    parser.add_argument("--port", type=int, default=7862, help="Port for the web server")
    parser.add_argument(
        "--omni-server",
        default=os.getenv("OMNI_SERVER_URL", DEFAULT_OMNI_SERVER_URL),
        help="Base URL for the vLLM-Omni server inside the container",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import uvicorn

    uvicorn.run(create_app(args.omni_server), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
