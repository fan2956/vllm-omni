const form = document.querySelector("#video-form");
const generateButton = document.querySelector("#generate-button");
const healthStatus = document.querySelector("#health-status");
const jobStatus = document.querySelector("#job-status");
const jobId = document.querySelector("#job-id");
const elapsed = document.querySelector("#elapsed");
const progress = document.querySelector("#progress");
const videoPlayer = document.querySelector("#video-player");
const message = document.querySelector("#message");

let activePoll = null;
let startedAt = null;

function setMessage(value, isError = false) {
  message.textContent = value || "";
  message.classList.toggle("error", isError);
}

function setStatus(status) {
  jobStatus.textContent = status;
  jobStatus.dataset.status = status;
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function payloadFromForm(formData) {
  const prompt = String(formData.get("prompt") || "").trim();
  return {
    server_id: "default",
    prompt,
    size: String(formData.get("size") || "720x1280").trim(),
    fps: numberOrNull(formData.get("fps")),
    num_frames: numberOrNull(formData.get("num_frames")),
    guidance_scale: numberOrNull(formData.get("guidance_scale")),
    flow_shift: numberOrNull(formData.get("flow_shift")),
    num_inference_steps: numberOrNull(formData.get("num_inference_steps")),
    seed: numberOrNull(formData.get("seed")),
    negative_prompt: String(formData.get("negative_prompt") || "").trim() || null,
    enable_frame_interpolation: formData.get("enable_frame_interpolation") === "on",
    frame_interpolation_model_path: String(formData.get("frame_interpolation_model_path") || "").trim() || null,
    frame_interpolation_exp: numberOrNull(formData.get("frame_interpolation_exp")),
    frame_interpolation_scale: numberOrNull(formData.get("frame_interpolation_scale")),
  };
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || payload;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail, null, 2));
  }
  return payload;
}

function updateElapsed() {
  if (!startedAt) {
    elapsed.textContent = "0s";
    return;
  }
  elapsed.textContent = `${Math.floor((Date.now() - startedAt) / 1000)}s`;
}

function stopPolling() {
  if (activePoll) {
    clearInterval(activePoll);
    activePoll = null;
  }
}

async function pollJob(id) {
  const data = await requestJson(`/api/videos/${id}`);
  setStatus(data.status || "unknown");
  progress.value = Number.isFinite(data.progress) ? data.progress : 0;
  updateElapsed();

  if (data.status === "completed") {
    stopPolling();
    generateButton.disabled = false;
    progress.value = 100;
    videoPlayer.src = `/api/videos/${id}/content?t=${Date.now()}`;
    videoPlayer.load();
    setMessage(`Completed in ${data.inference_time_s ? data.inference_time_s.toFixed(2) : "unknown"} seconds.`);
  } else if (data.status === "failed") {
    stopPolling();
    generateButton.disabled = false;
    const error = data.error?.message || "Video generation failed.";
    setMessage(error, true);
  }
}

async function checkHealth() {
  try {
    const data = await requestJson("/api/health");
    healthStatus.textContent = data.omni?.ok ? "Omni server connected" : "Omni server unavailable";
    healthStatus.classList.toggle("bad", !data.omni?.ok);
  } catch (error) {
    healthStatus.textContent = "Health check failed";
    healthStatus.classList.add("bad");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopPolling();
  setMessage("");
  setStatus("submitting");
  jobId.textContent = "-";
  progress.value = 0;
  videoPlayer.removeAttribute("src");
  videoPlayer.load();

  const payload = payloadFromForm(new FormData(form));
  if (!payload.prompt) {
    setStatus("idle");
    setMessage("Prompt is required.", true);
    return;
  }

  generateButton.disabled = true;
  startedAt = Date.now();
  updateElapsed();

  try {
    const data = await requestJson("/api/videos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    jobId.textContent = data.id || "-";
    setStatus(data.status || "queued");
    activePoll = setInterval(() => {
      pollJob(data.id).catch((error) => {
        stopPolling();
        generateButton.disabled = false;
        setStatus("error");
        setMessage(error.message, true);
      });
    }, 2000);
    await pollJob(data.id);
  } catch (error) {
    generateButton.disabled = false;
    setStatus("error");
    setMessage(error.message, true);
  }
});

checkHealth();
setInterval(checkHealth, 10000);
