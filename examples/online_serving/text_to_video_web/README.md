# Wan2.2 Text-to-Video Web Demo

This example provides a small browser UI for the asynchronous vLLM-Omni
`/v1/videos` API. It is intended to run inside the same Docker container as the
Wan2.2 vLLM-Omni server, while the page is opened from a browser on the host.

## Network Layout

- vLLM-Omni server runs in the container on `8099`.
- This web demo runs in the same container on `7862`.
- The host browser opens the mapped web port, for example `http://localhost:7862`.
- The browser never reads `VLLM_OMNI_STORAGE_PATH` directly. Video playback is
  proxied through `/api/videos/{video_id}/content`, which reads from the omni
  server's `/v1/videos/{video_id}/content` endpoint.

When using bridge networking, map both ports:

```bash
docker run --gpus all -p 8099:8099 -p 7862:7862 ...
```

If the container is started with `--net=host`, no `-p` mapping is needed.

## Start

Start the Wan2.2 server inside the container:

```bash
bash run.sh
```

In another shell inside the same container, start the web demo:

```bash
python examples/online_serving/text_to_video_web/app.py \
  --host 0.0.0.0 \
  --port 7862 \
  --omni-server http://127.0.0.1:8099
```

Open the page from the host browser:

```text
http://localhost:7862
```

If your container image does not already include the web dependencies, install
`fastapi`, `uvicorn`, and `httpx` in the same Python environment first.

You can also configure the omni server with an environment variable:

```bash
OMNI_SERVER_URL=http://127.0.0.1:8099 python examples/online_serving/text_to_video_web/app.py
```

## Defaults

The UI defaults match the local `curl.sh` example:

| Field | Default |
| --- | --- |
| `size` | `832x480`; selectable: `832x480`, `1280x720` |
| `fps` | `12` |
| `num_frames` | `61` |
| `guidance_scale` | `1.0` |
| `num_inference_steps` | `40` |
| `seed` | `42` |
| `enable_frame_interpolation` | `true` |

`flow_shift` is fixed to `5.0` in the proxy. When frame interpolation is
enabled, the proxy forwards the local RIFE defaults from the original curl
example: model path `/home/zf/vllm-omni/elfgum`, exp `1`, and scale `1.0`.
The UI only exposes the enable switch.

The job detail response includes `step_progress` when the omni server reports
real Wan2.2 denoising step callbacks. The page only displays server-reported
progress; it does not estimate in-flight progress locally.

## API Shape

The browser calls the local web backend:

- `GET /api/health`: check web and omni connectivity
- `POST /api/videos`: create a video generation job
- `GET /api/videos/{video_id}`: retrieve job status
- `GET /api/videos/{video_id}/content`: stream the generated MP4

`POST /api/videos` includes `server_id: "default"`. The value is fixed today,
but keeping it in the API lets a later two-server comparison view add named
server configs without changing the frontend contract.
