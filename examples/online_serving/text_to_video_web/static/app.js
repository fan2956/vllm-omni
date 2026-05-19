const form = document.querySelector("#video-form");
const generateButton = document.querySelector("#generate-button");
const healthStatus = document.querySelector("#health-status");
const compareEnabled = document.querySelector("#compare-enabled");
const compareToolbar = document.querySelector("#compare-toolbar");
const comparePlayButton = document.querySelector("#compare-play-button");
const comparePlayMessage = document.querySelector("#compare-play-message");

const resultViews = new Map(
  Array.from(document.querySelectorAll(".result-card")).map((panel) => [
    panel.dataset.serverId,
    {
      panel,
      latency: panel.querySelector(".latency"),
      progress: panel.querySelector(".progress"),
      stepProgress: panel.querySelector(".step-progress"),
      stepSpeed: panel.querySelector(".step-speed"),
      progressPercent: panel.querySelector(".progress-percent"),
      videoPlayer: panel.querySelector(".video-player"),
      message: panel.querySelector(".message"),
    },
  ]),
);

const activePolls = new Map();
const jobStates = new Map();

function isVideoReady(serverId) {
  return resultViews.get(serverId)?.panel.classList.contains("has-video") || false;
}

function updateComparePlaybackState(message = "") {
  const compareMode = compareEnabled.checked;
  const canCompare = compareMode && isVideoReady("default") && isVideoReady("compare");
  compareToolbar.hidden = !compareMode;
  comparePlayButton.disabled = !canCompare;
  comparePlayMessage.textContent = compareMode ? message : "";
}

function setMessage(view, value, isError = false) {
  view.message.textContent = value || "";
  view.message.classList.toggle("error", isError);
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function payloadFromForm(formData, serverId) {
  const prompt = String(formData.get("prompt") || "").trim();
  return {
    server_id: serverId,
    prompt,
    size: String(formData.get("size") || "832x480").trim(),
    fps: numberOrNull(formData.get("fps")),
    num_frames: numberOrNull(formData.get("num_frames")),
    guidance_scale: numberOrNull(formData.get("guidance_scale")),
    num_inference_steps: numberOrNull(formData.get("num_inference_steps")),
    seed: numberOrNull(formData.get("seed")),
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

function formatSeconds(seconds) {
  if (!Number.isFinite(seconds)) {
    return "0s";
  }
  return seconds < 10 ? `${seconds.toFixed(2)}s` : `${Math.round(seconds)}s`;
}

function formatStepSpeed(secondsPerStep) {
  if (!Number.isFinite(secondsPerStep) || secondsPerStep <= 0) {
    return "--";
  }
  return `${secondsPerStep.toFixed(2)}s/it`;
}

function stopPolling(serverId) {
  const poll = activePolls.get(serverId);
  if (poll) {
    clearInterval(poll);
    activePolls.delete(serverId);
  }
  const state = jobStates.get(serverId);
  if (state?.resolve) {
    state.resolve();
  }
}

function stopAllPolling() {
  for (const serverId of Array.from(activePolls.keys())) {
    stopPolling(serverId);
  }
}

function updateLatency(serverId, data = {}) {
  const view = resultViews.get(serverId);
  const state = jobStates.get(serverId);
  if (!view || !state?.startedAt) {
    return;
  }
  if (data.status === "completed" && Number.isFinite(data.inference_time_s)) {
    view.latency.textContent = formatSeconds(data.inference_time_s);
    return;
  }
  view.latency.textContent = formatSeconds((Date.now() - state.startedAt) / 1000);
}

function updateProgress(serverId, data, fallbackTotalSteps = 40) {
  const view = resultViews.get(serverId);
  if (!view) {
    return;
  }

  const stepData = data.step_progress || {};
  const totalSteps = Number.isFinite(stepData.total_steps) ? stepData.total_steps : fallbackTotalSteps;
  const currentStep = Number.isFinite(stepData.current_step) ? stepData.current_step : 0;
  const percent = Number.isFinite(stepData.percent)
    ? stepData.percent
    : Number.isFinite(data.progress)
      ? data.progress
      : 0;

  view.progress.value = Math.max(0, Math.min(100, percent));
  view.progressPercent.textContent = `${Math.round(view.progress.value)}%`;
  view.stepSpeed.textContent = formatStepSpeed(stepData.seconds_per_step);
  if (currentStep >= totalSteps && data.status === "in_progress") {
    view.stepProgress.textContent = `Step ${Math.max(0, currentStep)} / ${totalSteps} finalizing`;
  } else {
    view.stepProgress.textContent = `Step ${Math.max(0, currentStep)} / ${totalSteps}`;
  }
}

function resetResult(serverId, fallbackTotalSteps = 40) {
  const view = resultViews.get(serverId);
  if (!view) {
    return;
  }
  stopPolling(serverId);
  jobStates.delete(serverId);
  view.latency.textContent = "0s";
  updateProgress(serverId, {
    progress: 0,
    step_progress: { current_step: 0, total_steps: fallbackTotalSteps, percent: 0 },
  }, fallbackTotalSteps);
  view.panel.classList.remove("has-video");
  view.videoPlayer.removeAttribute("src");
  view.videoPlayer.load();
  setMessage(view, "");
  updateComparePlaybackState();
}

async function pollJob(serverId, fallbackTotalSteps = 40) {
  const state = jobStates.get(serverId);
  if (!state?.id) {
    return;
  }
  const view = resultViews.get(serverId);
  const data = await requestJson(`/api/videos/${state.id}?server_id=${encodeURIComponent(serverId)}`);
  updateProgress(serverId, data, fallbackTotalSteps);
  updateLatency(serverId, data);

  if (data.status === "completed") {
    stopPolling(serverId);
    updateProgress(serverId, data, fallbackTotalSteps);
    view.panel.classList.add("has-video");
    view.videoPlayer.src = `/api/videos/${state.id}/content?server_id=${encodeURIComponent(serverId)}&t=${Date.now()}`;
    view.videoPlayer.load();
    view.videoPlayer.play().catch(() => {});
    setMessage(view, "");
    updateComparePlaybackState();
  } else if (data.status === "failed") {
    stopPolling(serverId);
    const error = data.error?.message || "Video generation failed.";
    setMessage(view, error, true);
    updateComparePlaybackState();
  }
}

async function createAndPoll(serverId, payload, fallbackTotalSteps) {
  return new Promise(async (resolve) => {
    const view = resultViews.get(serverId);
    const startedAt = Date.now();
    jobStates.set(serverId, { startedAt, id: null, resolve });
    updateLatency(serverId);
    setMessage(view, "Submitting...");

    try {
      const data = await requestJson("/api/videos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      jobStates.set(serverId, { startedAt, id: data.id, resolve });
      updateProgress(serverId, data, fallbackTotalSteps);
      updateLatency(serverId, data);
      setMessage(view, "");

      activePolls.set(
        serverId,
        setInterval(() => {
          pollJob(serverId, fallbackTotalSteps).catch((error) => {
            stopPolling(serverId);
            updateLatency(serverId);
            setMessage(view, error.message, true);
          });
        }, 1000),
      );
      await pollJob(serverId, fallbackTotalSteps);
    } catch (error) {
      stopPolling(serverId);
      updateLatency(serverId);
      setMessage(view, error.message, true);
    }
  });
}

async function checkHealth() {
  try {
    const data = await requestJson(`/api/health?include_compare=${compareEnabled.checked ? "true" : "false"}`);
    const servers = data.servers || {};
    const defaultOk = servers.default?.ok || data.omni?.ok;
    const compareOk = servers.compare?.ok;
    if (compareEnabled.checked) {
      const unavailable = [];
      if (!defaultOk) {
        unavailable.push("Primary");
      }
      if (!compareOk) {
        unavailable.push("Compare");
      }
      healthStatus.textContent = defaultOk && compareOk
        ? "Both omni servers connected"
        : `${unavailable.join(" and ")} omni server unavailable`;
      healthStatus.classList.toggle("bad", !(defaultOk && compareOk));
    } else {
      healthStatus.textContent = defaultOk ? "Primary omni server connected" : "Primary omni server unavailable";
      healthStatus.classList.toggle("bad", !defaultOk);
    }
  } catch (error) {
    healthStatus.textContent = "Health check failed";
    healthStatus.classList.add("bad");
  }
}

function syncCompareVisibility() {
  const compareView = resultViews.get("compare");
  compareView.panel.hidden = !compareEnabled.checked;
  if (!compareEnabled.checked) {
    resetResult("compare");
  }
  updateComparePlaybackState();
  checkHealth();
}

async function playBothFromStart() {
  if (!isVideoReady("default") || !isVideoReady("compare")) {
    updateComparePlaybackState("Both videos must be ready.");
    return;
  }

  const players = [
    resultViews.get("default").videoPlayer,
    resultViews.get("compare").videoPlayer,
  ];
  comparePlayMessage.textContent = "";
  for (const player of players) {
    player.pause();
    player.currentTime = 0;
  }
  await Promise.allSettled(players.map((player) => player.play()));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopAllPolling();
  const formData = new FormData(form);
  const primaryPayload = payloadFromForm(formData, "default");
  const fallbackTotalSteps = primaryPayload.num_inference_steps || 40;

  resetResult("default", fallbackTotalSteps);
  resetResult("compare", fallbackTotalSteps);
  syncCompareVisibility();

  if (!primaryPayload.prompt) {
    setMessage(resultViews.get("default"), "Prompt is required.", true);
    return;
  }

  generateButton.disabled = true;
  const jobs = [createAndPoll("default", primaryPayload, fallbackTotalSteps)];
  if (compareEnabled.checked) {
    jobs.push(createAndPoll("compare", payloadFromForm(formData, "compare"), fallbackTotalSteps));
  }

  await Promise.allSettled(jobs);
  generateButton.disabled = false;
});

compareEnabled.addEventListener("change", syncCompareVisibility);
comparePlayButton.addEventListener("click", playBothFromStart);

resetResult("default");
resetResult("compare");
syncCompareVisibility();
setInterval(checkHealth, 10000);
