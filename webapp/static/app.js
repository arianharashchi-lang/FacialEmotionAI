const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusText = document.getElementById("status");
let stream = null;
let facingMode = "user";

async function loadEngine() {
  statusText.textContent = "Loading AI engine...";
  const response = await fetch("/api/start", { method: "POST" });
  if (!response.ok) {
    statusText.textContent = "Engine failed to load.";
    return false;
  }
  statusText.textContent = "AI engine ready.";
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
  overlay.textContent = "Camera ready. Live inference client will be enabled after local connectivity testing.";
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
  if (!stream) {
    await startCamera();
  }
  if (!stream) return;

  const analyzeButton = document.getElementById("analyze");
  analyzeButton.disabled = true;
  statusText.textContent = "Collecting a 20-frame temporal sequence...";

  await fetch("/api/reset", { method: "POST" });

  try {
    for (let i = 0; i < 20; i += 1) {
      const result = await captureAndAnalyzeOnce();
      updateAnalysis(result);
      await sleep(180);
    }
    statusText.textContent = "Temporal analysis complete.";
  } finally {
    analyzeButton.disabled = false;
  }
}

document.getElementById("analyze").addEventListener("click", analyzeBurst);
renderBars({});
