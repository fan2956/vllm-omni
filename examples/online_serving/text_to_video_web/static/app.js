const form = document.querySelector("#video-form");
const generateButton = document.querySelector("#generate-button");
const loopButton = document.querySelector("#loop-button");
const loopButtonText = loopButton.querySelector("span:last-child");
const loopStatus = document.querySelector("#loop-status");
const healthStatus = document.querySelector("#health-status");
const compareEnabled = document.querySelector("#compare-enabled");
const comparePlayButton = document.querySelector("#compare-play-button");
const comparePlayMessage = document.querySelector("#compare-play-message");
const promptInput = document.querySelector("#prompt");
const promptExampleButtons = document.querySelectorAll("[data-prompt]");
const accelerationCard = document.querySelector("#acceleration-card");
const accelerationRatio = document.querySelector("#acceleration-ratio");
const baselineLatency = document.querySelector("#baseline-latency");
const mindieLatency = document.querySelector("#mindie-latency");
const baselineBar = document.querySelector("#baseline-bar");
const mindieBar = document.querySelector("#mindie-bar");

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
const latencyState = new Map();
const loopState = {
  prompts: [],
  index: 0,
  running: false,
  stopRequested: false,
  activeGenerationToken: null,
};

function isVideoReady(serverId) {
  return resultViews.get(serverId)?.panel.classList.contains("has-video") || false;
}

function updateComparePlaybackState(message = "") {
  const compareMode = compareEnabled.checked;
  const canCompare = compareMode && isVideoReady("default") && isVideoReady("compare");
  comparePlayButton.hidden = !compareMode;
  comparePlayButton.disabled = !canCompare;
  comparePlayMessage.textContent = compareMode ? message : "";
}

function updateLoopButtonState() {
  loopButton.disabled = !loopState.running && loopState.prompts.length === 0;
  loopButton.classList.toggle("is-stopping", loopState.running && loopState.stopRequested);
  loopButtonText.textContent = loopState.running ? "STOP" : "LOOP";
}

function setLoopStatus(value) {
  loopStatus.textContent = value || "";
}

function setBarWidth(element, percent) {
  element.style.width = `${Math.max(0, Math.min(100, percent))}%`;
}

function updateAccelerationAnalysis() {
  const compareMode = compareEnabled.checked;
  accelerationCard.hidden = !compareMode;
  if (!compareMode) {
    return;
  }

  const acceleratedSeconds = latencyState.get("default");
  const baselineSeconds = latencyState.get("compare");
  const hasAccelerated = Number.isFinite(acceleratedSeconds) && acceleratedSeconds > 0;
  const hasBaseline = Number.isFinite(baselineSeconds) && baselineSeconds > 0;

  baselineLatency.textContent = hasBaseline ? formatSeconds(baselineSeconds) : "--";
  mindieLatency.textContent = hasAccelerated ? formatSeconds(acceleratedSeconds) : "--";
  setBarWidth(baselineBar, hasBaseline ? 100 : 0);

  if (hasAccelerated && hasBaseline) {
    const ratio = baselineSeconds / acceleratedSeconds;
    accelerationRatio.textContent = `${ratio.toFixed(2)}x`;
    setBarWidth(mindieBar, Math.max(8, (acceleratedSeconds / baselineSeconds) * 100));
  } else {
    accelerationRatio.textContent = "--";
    setBarWidth(mindieBar, hasAccelerated ? 100 : 0);
  }
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

function randomSeed() {
  const maxSeed = 2147483647;
  if (window.crypto?.getRandomValues) {
    const values = new Uint32Array(1);
    window.crypto.getRandomValues(values);
    return (values[0] % maxSeed) + 1;
  }
  return Math.floor(Math.random() * maxSeed) + 1;
}

function sharedSeedForComparison(formData) {
  const seedValue = String(formData.get("seed") ?? "").trim();
  if (!seedValue || seedValue === "-1") {
    return randomSeed();
  }
  const parsed = Number(seedValue);
  return Number.isInteger(parsed) ? parsed : randomSeed();
}

function payloadFromForm(formData, serverId, seedOverride) {
  const prompt = String(formData.get("prompt") || "").trim();
  return {
    server_id: serverId,
    prompt,
    size: String(formData.get("size") || "832x480").trim(),
    fps: numberOrNull(formData.get("fps")),
    num_frames: numberOrNull(formData.get("num_frames")),
    guidance_scale: numberOrNull(formData.get("guidance_scale")),
    num_inference_steps: numberOrNull(formData.get("num_inference_steps")),
    seed: seedOverride ?? numberOrNull(formData.get("seed")),
    enable_frame_interpolation: formData.get("enable_frame_interpolation") === "on",
  };
}

async function loadLoopPrompts() {
  try {
    const data = await requestJson("/api/prompts");
    if (!data.ok) {
      loopState.prompts = [];
      setLoopStatus(data.detail || "Prompt file is not configured.");
      updateLoopButtonState();
      return;
    }
    loopState.prompts = Array.isArray(data.prompts) ? data.prompts : [];
    loopState.index = 0;
    setLoopStatus(loopState.prompts.length > 0
      ? `Prompts ${loopState.prompts.length} lines`
      : "Prompt file has no non-empty lines.");
  } catch (error) {
    loopState.prompts = [];
    setLoopStatus(error.message);
  }
  updateLoopButtonState();
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

function resetJobState(serverId, fallbackTotalSteps = 40, { clearVideo = true } = {}) {
  const view = resultViews.get(serverId);
  if (!view) {
    return;
  }
  stopPolling(serverId);
  jobStates.delete(serverId);
  latencyState.delete(serverId);
  view.latency.textContent = "0s";
  updateProgress(serverId, {
    progress: 0,
    step_progress: { current_step: 0, total_steps: fallbackTotalSteps, percent: 0 },
  }, fallbackTotalSteps);
  if (clearVideo) {
    view.panel.classList.remove("has-video");
    view.videoPlayer.loop = false;
    view.videoPlayer.removeAttribute("src");
    view.videoPlayer.load();
  }
  setMessage(view, "");
  updateComparePlaybackState();
  updateAccelerationAnalysis();
}

function updateLatency(serverId, data = {}) {
  const view = resultViews.get(serverId);
  const state = jobStates.get(serverId);
  if (!view || !state?.startedAt) {
    return;
  }
  let seconds;
  if (data.status === "completed" && Number.isFinite(data.inference_time_s)) {
    seconds = data.inference_time_s;
    view.latency.textContent = formatSeconds(seconds);
    latencyState.set(serverId, seconds);
    updateAccelerationAnalysis();
    return;
  }
  seconds = (Date.now() - state.startedAt) / 1000;
  view.latency.textContent = formatSeconds(seconds);
  latencyState.set(serverId, seconds);
  updateAccelerationAnalysis();
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
  resetJobState(serverId, fallbackTotalSteps, { clearVideo: true });
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

async function createAndPollLoopJob(serverId, payload, fallbackTotalSteps, generationToken) {
  const view = resultViews.get(serverId);
  const startedAt = Date.now();
  jobStates.set(serverId, { startedAt, id: null });
  updateLatency(serverId);
  setMessage(view, "Submitting...");

  try {
    const created = await requestJson("/api/videos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    jobStates.set(serverId, { startedAt, id: created.id });
    updateProgress(serverId, created, fallbackTotalSteps);
    updateLatency(serverId, created);
    setMessage(view, "");

    while (!loopState.stopRequested && loopState.activeGenerationToken === generationToken) {
      const data = await requestJson(`/api/videos/${created.id}?server_id=${encodeURIComponent(serverId)}`);
      updateProgress(serverId, data, fallbackTotalSteps);
      updateLatency(serverId, data);

      if (data.status === "completed") {
        updateProgress(serverId, data, fallbackTotalSteps);
        setMessage(view, "");
        return {
          ok: true,
          serverId,
          contentUrl: `/api/videos/${created.id}/content?server_id=${encodeURIComponent(serverId)}&t=${Date.now()}`,
        };
      }
      if (data.status === "failed") {
        const error = data.error?.message || "Video generation failed.";
        setMessage(view, error, true);
        return { ok: false, serverId, error };
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    return { ok: false, serverId, stopped: true };
  } catch (error) {
    updateLatency(serverId);
    setMessage(view, error.message, true);
    return { ok: false, serverId, error: error.message };
  }
}

async function generateLoopRound(prompt, promptIndex, generationToken) {
  promptInput.value = prompt;
  const formData = new FormData(form);
  const compareMode = compareEnabled.checked;
  const compareSeed = compareMode ? sharedSeedForComparison(formData) : undefined;
  const primaryPayload = payloadFromForm(formData, "default", compareSeed);
  const fallbackTotalSteps = primaryPayload.num_inference_steps || 40;

  resetJobState("default", fallbackTotalSteps, { clearVideo: !isVideoReady("default") });
  resetJobState("compare", fallbackTotalSteps, { clearVideo: !isVideoReady("compare") });
  syncCompareVisibility();

  if (!primaryPayload.prompt) {
    setMessage(resultViews.get("default"), "Prompt is required.", true);
    return null;
  }

  const nextIndex = (promptIndex % loopState.prompts.length) + 1;
  setLoopStatus(`Playing current, generating ${nextIndex}/${loopState.prompts.length}`);
  const jobs = [createAndPollLoopJob("default", primaryPayload, fallbackTotalSteps, generationToken)];
  if (compareMode) {
    jobs.push(createAndPollLoopJob("compare", payloadFromForm(formData, "compare", compareSeed), fallbackTotalSteps, generationToken));
  }

  const results = await Promise.all(jobs);
  if (loopState.stopRequested || loopState.activeGenerationToken !== generationToken) {
    return null;
  }
  if (!results.every((result) => result.ok)) {
    return null;
  }
  return { compareMode, results, promptIndex };
}

async function runGeneration(formData) {
  stopAllPolling();
  const compareMode = compareEnabled.checked;
  const compareSeed = compareMode ? sharedSeedForComparison(formData) : undefined;
  const primaryPayload = payloadFromForm(formData, "default", compareSeed);
  const fallbackTotalSteps = primaryPayload.num_inference_steps || 40;

  resetResult("default", fallbackTotalSteps);
  resetResult("compare", fallbackTotalSteps);
  syncCompareVisibility();

  if (!primaryPayload.prompt) {
    setMessage(resultViews.get("default"), "Prompt is required.", true);
    return false;
  }

  const jobs = [createAndPoll("default", primaryPayload, fallbackTotalSteps)];
  if (compareMode) {
    jobs.push(createAndPoll("compare", payloadFromForm(formData, "compare", compareSeed), fallbackTotalSteps));
  }

  await Promise.allSettled(jobs);
  return true;
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
  updateAccelerationAnalysis();
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

async function swapLoopVideos(loopResult) {
  if (!loopResult) {
    return;
  }

  const readyPlayers = [];
  for (const result of loopResult.results) {
    const view = resultViews.get(result.serverId);
    if (!view) {
      continue;
    }
    view.panel.classList.add("has-video");
    view.videoPlayer.src = result.contentUrl;
    view.videoPlayer.load();
    readyPlayers.push(view.videoPlayer);
  }
  if (readyPlayers.length === 0) {
    return;
  }

  for (const player of readyPlayers) {
    player.pause();
    player.muted = true;
    player.loop = true;
    player.currentTime = 0;
  }
  await Promise.allSettled(readyPlayers.map((player) => player.play()));
  updateComparePlaybackState();
  const currentIndex = loopResult.promptIndex + 1;
  setLoopStatus(`Playing ${currentIndex}/${loopState.prompts.length}`);
}

async function runPromptLoop() {
  if (loopState.running) {
    loopState.stopRequested = true;
    setLoopStatus("STOP requested; current video will keep playing.");
    updateLoopButtonState();
    return;
  }
  if (loopState.prompts.length === 0) {
    setLoopStatus("Prompt file has no available prompts.");
    updateLoopButtonState();
    return;
  }

  loopState.running = true;
  loopState.stopRequested = false;
  loopState.activeGenerationToken = Symbol("loop-generation");
  updateLoopButtonState();

  while (!loopState.stopRequested && loopState.prompts.length > 0) {
    const generationToken = loopState.activeGenerationToken;
    const prompt = loopState.prompts[loopState.index];
    const loopResult = await generateLoopRound(prompt, loopState.index, generationToken);
    if (loopState.stopRequested || loopState.activeGenerationToken !== generationToken) {
      break;
    }
    if (loopResult) {
      await swapLoopVideos(loopResult);
    }
    loopState.index = (loopState.index + 1) % loopState.prompts.length;
  }

  loopState.running = false;
  loopState.stopRequested = false;
  loopState.activeGenerationToken = null;
  setLoopStatus(loopState.prompts.length > 0 ? `Stopped at ${loopState.index + 1}/${loopState.prompts.length}` : "");
  updateLoopButtonState();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (loopState.running) {
    loopState.stopRequested = true;
    setLoopStatus("STOP requested; current video will keep playing.");
    updateLoopButtonState();
    return;
  }

  generateButton.disabled = true;
  await runGeneration(new FormData(form));
  generateButton.disabled = false;
});

compareEnabled.addEventListener("change", syncCompareVisibility);
comparePlayButton.addEventListener("click", playBothFromStart);
loopButton.addEventListener("click", runPromptLoop);
for (const button of promptExampleButtons) {
  button.addEventListener("click", () => {
    promptInput.value = button.dataset.prompt || "";
    promptInput.focus();
  });
}

resetResult("default");
resetResult("compare");
syncCompareVisibility();
loadLoopPrompts();
setInterval(checkHealth, 10000);
