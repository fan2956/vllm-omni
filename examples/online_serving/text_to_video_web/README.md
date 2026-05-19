# Wan2.2 Text-to-Video Web Demo

This example provides a small browser UI for the asynchronous vLLM-Omni
`/v1/videos` API. It is intended to run inside the same Docker container as the
Wan2.2 vLLM-Omni server, while the page is opened from a browser on the host.
It can also submit the same prompt to a second server for side-by-side latency
and video comparison.

## Network Layout

- Primary vLLM-Omni server runs in the container on `8099`.
- Optional comparison vLLM-Omni server runs in the container on `9099`.
- This web demo runs in the same container on `7862`.
- The host browser opens the mapped web port, for example `http://localhost:7862`.
- The browser never reads `VLLM_OMNI_STORAGE_PATH` directly. Video playback is
  proxied through `/api/videos/{video_id}/content`, which reads from the omni
  server's `/v1/videos/{video_id}/content` endpoint.

When using bridge networking, map all serving ports:

```bash
docker run --gpus all -p 8099:8099 -p 9099:9099 -p 7862:7862 ...
```

If the container is started with `--net=host`, no `-p` mapping is needed.

## Start

Start the Wan2.2 server inside the container. For comparison mode, start two
vLLM-Omni servers, one on `8099` and one on `9099`.

```bash
bash run.sh
```

In another shell inside the same container, start the web demo:

```bash
python examples/online_serving/text_to_video_web/app.py \
  --host 0.0.0.0 \
  --port 7862 \
  --omni-server http://127.0.0.1:8099 \
  --compare-omni-server http://127.0.0.1:9099
```

Open the page from the host browser:

```text
http://localhost:7862
```

If your container image does not already include the web dependencies, install
`fastapi`, `uvicorn`, and `httpx` in the same Python environment first.

You can also configure the omni servers with environment variables:

```bash
OMNI_SERVER_URL=http://127.0.0.1:8099 \
OMNI_COMPARE_SERVER_URL=http://127.0.0.1:9099 \
python examples/online_serving/text_to_video_web/app.py
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
| `seed` | `-1`, which means no fixed seed |
| `negative_prompt` | Hidden Chinese Wan2.2 quality/detail negative prompt |
| `enable_frame_interpolation` | `true` |
| `compare_enabled` | `true` |

`flow_shift` is fixed to `5.0` in the proxy. When frame interpolation is
enabled, the proxy forwards the local RIFE defaults from the original curl
example: model path `/home/zf/vllm-omni/elfgum`, exp `1`, and scale `1.0`.
The UI only exposes the enable switch.

The proxy keeps a default negative prompt even though it is not shown on the
page. Seed `-1` or an omitted seed is not forwarded to vLLM-Omni, so generation
will not use a fixed seed unless the user enters one.

The job detail response includes `step_progress` when the omni server reports
real Wan2.2 denoising step callbacks. The page only displays server-reported
progress; it does not estimate in-flight progress locally.

## API Shape

The browser calls the local web backend:

- `GET /api/health`: check web and omni connectivity; pass
  `include_compare=false` to check only `8099`
- `POST /api/videos`: create a video generation job
- `GET /api/videos/{video_id}?server_id=...`: retrieve job status
- `GET /api/videos/{video_id}/content?server_id=...`: stream the generated MP4

`POST /api/videos` includes `server_id`. Use `default` for the `8099` panel
(`vLLM-Omni (Speed By MindIE SD)`) and `compare` for the `9099` panel
(`vLLM-Omni`).
