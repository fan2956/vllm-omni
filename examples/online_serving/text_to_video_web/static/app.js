const form = document.querySelector("#video-form");
const generateButton = document.querySelector("#generate-button");
const healthStatus = document.querySelector("#health-status");
const jobStatus = document.querySelector("#job-status");
const jobId = document.querySelector("#job-id");
const elapsed = document.querySelector("#elapsed");
const progress = document.querySelector("#progress");
const stepProgress = document.querySelector("#step-progress");
const progressPercent = document.querySelector("#progress-percent");
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
    size: String(formData.get("size") || "832x480").trim(),
    fps: numberOrNull(formData.get("fps")),
    num_frames: numberOrNull(formData.get("num_frames")),
    guidance_scale: numberOrNull(formData.get("guidance_scale")),
    num_inference_steps: numberOrNull(formData.get("num_inference_steps")),
    seed: numberOrNull(formData.get("seed")),
    negative_prompt: String(formData.get("negative_prompt") || "").trim() || null,
    enable_frame_interpolation: formData.get("enable_frame_interpolation") === "on",
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

function updateProgress(data, fallbackTotalSteps = 40) {
  const stepData = data.step_progress || {};
  const totalSteps = Number.isFinite(stepData.total_steps) ? stepData.total_steps : fallbackTotalSteps;
  const currentStep = Number.isFinite(stepData.current_step) ? stepData.current_step : 0;
  const percent = Number.isFinite(stepData.percent)
    ? stepData.percent
    : Number.isFinite(data.progress)
      ? data.progress
      : 0;

  progress.value = Math.max(0, Math.min(100, percent));
  progressPercent.textContent = `${Math.round(progress.value)}%`;
  if (currentStep >= totalSteps && data.status === "in_progress") {
    stepProgress.textContent = `Step ${Math.max(0, currentStep)} / ${totalSteps} finalizing`;
  } else {
    stepProgress.textContent = `Step ${Math.max(0, currentStep)} / ${totalSteps}`;
  }
}

async function pollJob(id) {
  const data = await requestJson(`/api/videos/${id}`);
  setStatus(data.status || "unknown");
  updateProgress(data);
  updateElapsed();

  if (data.status === "completed") {
    stopPolling();
    generateButton.disabled = false;
    updateProgress(data);
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
  updateProgress({ progress: 0, step_progress: { current_step: 0, total_steps: 40, percent: 0 } });
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
    updateProgress(data, payload.num_inference_steps || 40);
    activePoll = setInterval(() => {
      pollJob(data.id).catch((error) => {
        stopPolling();
        generateButton.disabled = false;
        setStatus("error");
        setMessage(error.message, true);
      });
    }, 1000);
    await pollJob(data.id);
  } catch (error) {
    generateButton.disabled = false;
    setStatus("error");
    setMessage(error.message, true);
  }
});

updateProgress({ progress: 0, step_progress: { current_step: 0, total_steps: 40, percent: 0 } });
checkHealth();
setInterval(checkHealth, 10000);
