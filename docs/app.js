const API_BASE = "https://facialemotionai-web.onrender.com";
const backendBadge = document.getElementById("backendBadge");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusText = document.getElementById("status");
let stream = null;
let facingMode = "user";

async function loadEngine() {
  statusText.textContent = "Connecting to the AI engine...";
  try {
    const response = await fetch(`${API_BASE}/api/start`, { method: "POST" });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      statusText.textContent = `AI engine failed to load (HTTP ${response.status}). ${detail.slice(0, 160)}`;
      return false;
    }
    statusText.textContent = "AI engine ready.";
    if (backendBadge) {
      backendBadge.textContent = "Inference API online";
      backendBadge.classList.add("online");
      backendBadge.classList.remove("offline");
    }
    return true;
  } catch (error) {
    statusText.textContent = "Cannot reach the inference server. The public interface is online, but analysis needs the Render API to respond.";
    if (backendBadge) {
      backendBadge.textContent = "Inference API unreachable";
      backendBadge.classList.add("offline");
      backendBadge.classList.remove("online");
    }
    return false;
  }
}

async function startCamera() {
  try {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode },
      audio: false
    });
    video.srcObject = stream;
    overlay.textContent = "Camera preview ready. AI analysis requires the inference API to be online.";
    return true;
  } catch (error) {
    statusText.textContent = "Camera could not start: " + error.message;
    return false;
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  overlay.textContent = "Camera is off";
}

document.getElementById("load").addEventListener("click", loadEngine);
document.getElementById("start").addEventListener("click", startCamera);
document.getElementById("stop").addEventListener("click", stopCamera);
document.getElementById("switch").addEventListener("click", async () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  await startCamera();
});

window.addEventListener("beforeunload", stopCamera);


const canvas = document.getElementById("canvas");
const emotionText = document.getElementById("emotion");
const confidenceText = document.getElementById("confidence");
const reliabilityText = document.getElementById("reliability");
const bars = document.getElementById("bars");
const spoofText = document.getElementById("spoof");

function pct(value) {
  return Math.round((Number(value) || 0) * 100);
}

function renderBars(probabilities = {}) {
  const classes = ["angry", "happy", "surprise", "sad", "neutral"];
  bars.innerHTML = classes.map((name) => {
    const value = Number(probabilities[name] || 0);
    return '<div class="bar">' +
      '<div class="bar-head"><span>' + name + '</span><span>' + pct(value) + '%</span></div>' +
      '<div class="track"><div class="fill" style="width:' + pct(value) + '%"></div></div>' +
      '</div>';
  }).join("");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function captureAndAnalyzeOnce() {
  if (!stream || video.readyState < 2) return null;

  const width = 480;
  const ratio = video.videoWidth ? video.videoHeight / video.videoWidth : 0.75;
  canvas.width = width;
  canvas.height = Math.round(width * ratio);

  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.72));
  if (!blob) return null;

  const form = new FormData();
  form.append("frame", blob, "frame.jpg");

  const response = await fetch(`${API_BASE}/api/frame`, {
    method: "POST",
    body: form
  });

  if (!response.ok) return null;
  return await response.json();
}

function updateAnalysis(result) {
  if (!result) return;

  if (!result.face_detected) {
    overlay.textContent = "No face detected";
    return;
  }

  const anti = result.anti_spoof || {};
  spoofText.textContent = "Anti-spoofing: " + (anti.status || "—");

  if (anti.status === "FAKE") {
    overlay.textContent = "Frame rejected by anti-spoofing";
    return;
  }

  const emotion = result.emotion || {};
  if (!emotion.ready) {
    overlay.textContent = "Temporal warmup: " +
      (emotion.collected || 0) + " / " + (emotion.required || 20);
    return;
  }

  overlay.textContent = "";
  emotionText.textContent = emotion.label || "—";
  confidenceText.textContent = pct(emotion.confidence) + "% confidence";
  reliabilityText.textContent = emotion.reliability || "—";
  renderBars(emotion.probabilities || {});
}

async function analyzeBurst() {
  if (!(await loadEngine())) return;
  if (!stream) await startCamera();
  if (!stream) return;

  const analyzeButton = document.getElementById("analyze");
  analyzeButton.disabled = true;
  statusText.textContent = "Collecting a 20-frame temporal sequence...";

  try {
    const resetResponse = await fetch(`${API_BASE}/api/reset`, { method: "POST" });
    if (!resetResponse.ok) throw new Error("Could not reset the analysis sequence.");
    for (let i = 0; i < 20; i += 1) {
      const result = await captureAndAnalyzeOnce();
      updateAnalysis(result);
      await sleep(180);
    }
    statusText.textContent = "Temporal analysis complete.";
  } catch (error) {
    statusText.textContent = "Analysis failed: " + error.message;
  } finally {
    analyzeButton.disabled = false;
  }
}

document.getElementById("analyze").addEventListener("click", analyzeBurst);
renderBars({});


let latestVideoResult = null;

function renderVideoBars(probabilities = {}) {
  const target = document.getElementById("videoBars");
  const classes = ["angry", "happy", "surprise", "sad", "neutral"];
  target.innerHTML = classes.map((name) => {
    const value = Number(probabilities[name] || 0);
    return '<div class="bar">' +
      '<div class="bar-head"><span>' + name + '</span><span>' + pct(value) + '%</span></div>' +
      '<div class="track"><div class="fill" style="width:' + pct(value) + '%"></div></div>' +
      '</div>';
  }).join("");
}

function renderVideoResult(data) {
  const box = document.getElementById("videoResult");
  box.classList.remove("hidden");

  if (!data.summary) {
    document.getElementById("videoDominant").textContent = "No result";
    document.getElementById("videoConfidence").textContent = "—";
    document.getElementById("timeline").innerHTML = "<p>No ready temporal predictions were produced.</p>";
    return;
  }

  document.getElementById("videoDominant").textContent = data.summary.dominant_expression || "—";
  document.getElementById("videoConfidence").textContent = pct(data.summary.average_confidence) + "%";
  renderVideoBars(data.summary.average_probabilities || {});

  document.getElementById("timeline").innerHTML = (data.timeline || []).map((point) => {
    return '<div class="timeline-row">' +
      '<span>' + Number(point.time).toFixed(1) + 's</span>' +
      '<strong>' + point.label + '</strong>' +
      '<span>' + pct(point.confidence) + '% • ' + (point.reliability || "—") + '</span>' +
      '</div>';
  }).join("");
}

async function analyzeVideoFile() {
  const fileInput = document.getElementById("videoFile");
  const file = fileInput.files[0];
  const message = document.getElementById("videoStatus");
  const button = document.getElementById("analyzeVideo");

  if (!file) {
    message.textContent = "Choose a video first.";
    return;
  }

  if (!(await loadEngine())) return;

  stopCamera();
  button.disabled = true;
  message.textContent = "Analyzing video... keep this page open.";

  const form = new FormData();
  form.append("video", file);

  try {
    const response = await fetch(`${API_BASE}/api/video`, { method: "POST", body: form });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Video analysis failed.");
    }

    latestVideoResult = data;
    renderVideoResult(data);
    message.textContent = data.message || "Analysis complete.";
  } catch (error) {
    console.error(error);
    message.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function downloadVideoCsv() {
  if (!latestVideoResult || !Array.isArray(latestVideoResult.timeline)) return;

  const classes = ["angry", "happy", "surprise", "sad", "neutral"];
  const header = ["time_seconds", "label", "confidence", "reliability", ...classes];
  const rows = latestVideoResult.timeline.map((point) => {
    const p = point.probabilities || {};
    return [
      point.time,
      point.label,
      point.confidence,
      point.reliability,
      ...classes.map((name) => p[name] ?? 0)
    ];
  });

  const csv = [header, ...rows]
    .map((row) => row.map((value) => JSON.stringify(value ?? "")).join(","))
    .join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "facialemotionai-analysis.csv";
  anchor.click();
  URL.revokeObjectURL(url);
}

document.getElementById("analyzeVideo").addEventListener("click", analyzeVideoFile);
document.getElementById("downloadCsv").addEventListener("click", downloadVideoCsv);
document.getElementById("printReport").addEventListener("click", () => window.print());


async function checkBackend() {
  if (!backendBadge) return;
  try {
    const response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const data = await response.json();
    if (data.status !== "ok") throw new Error("API not healthy");
    backendBadge.textContent = "Inference API online";
    backendBadge.classList.add("online");
    backendBadge.classList.remove("offline");
    statusText.textContent = "Website loaded. Click Load AI Engine to initialize the models.";
  } catch {
    backendBadge.textContent = "Inference API unreachable";
    backendBadge.classList.add("offline");
    backendBadge.classList.remove("online");
    statusText.textContent = "Website is online, but the Render inference server is not responding. Camera preview works; analysis needs the API.";
  }
}
checkBackend();
