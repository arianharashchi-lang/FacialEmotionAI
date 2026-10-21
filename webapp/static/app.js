const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusText = document.getElementById("status");
let stream = null;
let facingMode = "user";
const tr = (key) => window.FacialEmotionI18n ? window.FacialEmotionI18n.t(key) : key;
const emotionName = (name) => {
  const keys = ["angry", "happy", "surprise", "sad", "neutral"];
  const i = keys.indexOf(String(name || "").toLowerCase());
  const d = window.FacialEmotionI18n?.languages[document.getElementById("languageSelect")?.value || "en"];
  return i >= 0 ? (d?.classes?.[i] || ["Angry", "Happy", "Surprise", "Sad", "Neutral"][i]) : (name || "—");
};

async function loadEngine() {
  statusText.textContent = tr("loading");
  const response = await fetch("/api/start", { method: "POST" });
  if (!response.ok) {
    statusText.textContent = tr("failed");
    return false;
  }
  statusText.textContent = tr("ready");
  return true;
}

async function startCamera() {
  if (!(await loadEngine())) return;

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode },
    audio: false
  });

  video.srcObject = stream;
  overlay.textContent = tr("cameraReady");
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  overlay.textContent = tr("cameraOff");
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
      '<div class="bar-head"><span>' + emotionName(name) + '</span><span>' + pct(value) + '%</span></div>' +
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

  const response = await fetch("/api/frame", {
    method: "POST",
    body: form
  });

  if (!response.ok) return null;
  return await response.json();
}

function updateAnalysis(result) {
  if (!result) return;

  if (!result.face_detected) {
    overlay.textContent = tr("noFace");
    return;
  }

  const anti = result.anti_spoof || {};
  spoofText.textContent = tr("spoof") + (anti.status || "—");

  if (anti.status === "FAKE") {
    overlay.textContent = tr("rejected");
    return;
  }

  const emotion = result.emotion || {};
  if (!emotion.ready) {
    overlay.textContent = tr("warmup") +
      (emotion.collected || 0) + " / " + (emotion.required || 20);
    return;
  }

  overlay.textContent = "";
  emotionText.textContent = emotion.label || "—";
  confidenceText.textContent = pct(emotion.confidence) + " " + tr("confidence");
  reliabilityText.textContent = emotion.reliability || "—";
  renderBars(emotion.probabilities || {});
}

async function analyzeBurst() {
  if (!stream) {
    await startCamera();
  }
  if (!stream) return;

  const analyzeButton = document.getElementById("analyze");
  analyzeButton.disabled = true;
  statusText.textContent = tr("collecting");

  await fetch("/api/reset", { method: "POST" });

  try {
    for (let i = 0; i < 20; i += 1) {
      const result = await captureAndAnalyzeOnce();
      updateAnalysis(result);
      await sleep(180);
    }
    statusText.textContent = tr("complete");
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
    document.getElementById("videoDominant").textContent = tr("noResult");
    document.getElementById("videoConfidence").textContent = "—";
    document.getElementById("timeline").innerHTML = "<p>No ready temporal predictions were produced.</p>";
    return;
  }

  document.getElementById("videoDominant").textContent = emotionName(data.summary.dominant_expression);
  document.getElementById("videoConfidence").textContent = pct(data.summary.average_confidence) + "%";
  renderVideoBars(data.summary.average_probabilities || {});

  document.getElementById("timeline").innerHTML = (data.timeline || []).map((point) => {
    return '<div class="timeline-row">' +
      '<span>' + Number(point.time).toFixed(1) + 's</span>' +
      '<strong>' + emotionName(point.label) + '</strong>' +
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
    message.textContent = tr("selectVideo");
    return;
  }

  if (!(await loadEngine())) return;

  stopCamera();
  button.disabled = true;
  message.textContent = tr("analyzing");

  const form = new FormData();
  form.append("video", file);

  try {
    const response = await fetch("/api/video", { method: "POST", body: form });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || tr("failed"));
    }

    latestVideoResult = data;
    renderVideoResult(data);
    message.textContent = data.message || tr("complete");
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


// Refresh translated dynamic labels when the language changes.
document.getElementById("languageSelect")?.addEventListener("change", () => {
  renderBars({});
  if (latestVideoResult) renderVideoResult(latestVideoResult);
});
