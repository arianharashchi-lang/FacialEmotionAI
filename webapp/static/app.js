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
