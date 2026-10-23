"use strict";
const $=id=>document.getElementById(id);
const video=$("camera");
let stream=null,modelsReady=false,busy=false,timer=null,descriptor=null,pinHash=null,eyesClosedSince=null,headAwaySince=null,yawnSince=null,liveDetection=null;
let virtualIgnition=false;
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const setText=(id,v)=>$(id).textContent=v;
function secureContext(){return window.isSecureContext&&!!navigator.mediaDevices?.getUserMedia;}
async function hashPin(value){const bytes=new TextEncoder().encode(value);const hash=await crypto.subtle.digest("SHA-256",bytes);return Array.from(new Uint8Array(hash)).map(x=>x.toString(16).padStart(2,"0")).join("");}
function setStatus(message){setText("status",message);}
async function loadModels(){
 if(modelsReady)return;
 if(!window.faceapi)throw Error("Face model library was blocked or unavailable (CDN).");
 const root="https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights";
 await Promise.all([faceapi.nets.tinyFaceDetector.loadFromUri(root),faceapi.nets.faceLandmark68Net.loadFromUri(root),faceapi.nets.faceRecognitionNet.loadFromUri(root)]);
 modelsReady=true;
}
const opts=()=>new faceapi.TinyFaceDetectorOptions({inputSize:320,scoreThreshold:.5});
async function detect(){if(!modelsReady||!stream||video.readyState<2)return null;return await faceapi.detectSingleFace(video,opts()).withFaceLandmarks().withFaceDescriptor();}
function ear(points){return (distance(points[1],points[5])+distance(points[2],points[4]))/(2*Math.max(.001,distance(points[0],points[3])));}
function monitoring(result){
 liveDetection=result;setText("faceState",result?"Detected":"Not detected");
 if(!result){["eyeState","yawnState","headState","fatigueState","distractionState"].forEach(x=>setText(x,"Unknown"));eyesClosedSince=headAwaySince=yawnSince=null;setText("monitorState","Tracking lost");return;}
 const p=result.landmarks.positions;
 const e=(ear(p.slice(36,42))+ear(p.slice(42,48)))/2;
 const mouth=distance(p[62],p[66])/Math.max(1,distance(p[60],p[64]));
 const eyeCenter=(p[36].x+p[45].x)/2,eyeWidth=distance(p[36],p[45]);
 const offset=(p[30].x-eyeCenter)/Math.max(1,eyeWidth);
 const closed=e<.21, yawning=mouth>.45, away=Math.abs(offset)>.25,now=performance.now();
 eyesClosedSince=closed?(eyesClosedSince??now):null;
 yawnSince=yawning?(yawnSince??now):null;
 headAwaySince=away?(headAwaySince??now):null;
 const fatigue=eyesClosedSince!==null&&now-eyesClosedSince>1500;
 const distracted=headAwaySince!==null&&now-headAwaySince>2000;
 setText("eyeState",closed?"Likely closed":"Open");setText("yawnState",yawning?"Mouth open":"No cue");
 setText("headState",away?"Turned away":"Centered");
 setText("fatigueState",fatigue?"Warning":"No sustained closure");
 setText("distractionState",distracted?"Warning":"No prolonged head turn");
 setText("monitorState",fatigue||distracted?"Attention required":"Monitoring");
}
async function cycle(){if(busy||!stream||!modelsReady)return;busy=true;try{monitoring(await detect());}catch(e){setStatus("Tracking error: "+e.message);}finally{busy=false;}}
async function start(){
 try{
 if(!secureContext())throw Error("Camera API unavailable. Use HTTPS/localhost and a supported browser.");
 setStatus("Requesting camera...");
 stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:"user"}});
 video.srcObject=stream;await video.play();
 setText("cameraMessage","Camera on");setStatus("Loading face models... first run requires a network connection.");
 await loadModels();setStatus("Models ready. Enroll an authorized driver or monitor facial cues.");
 $("enroll").disabled=false;$("verify").disabled=false;
 clearInterval(timer);timer=setInterval(cycle,650);
 }catch(e){setStatus("Could not start: "+e.message);stop();}
}
function stop(){clearInterval(timer);timer=null;if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}video.srcObject=null;setText("cameraMessage","Camera off");$("enroll").disabled=true;$("verify").disabled=true;monitoring(null);setText("monitorState","Inactive");}
async function enroll(){
 const pin=$("pin").value;
 if(pin.length<6){setText("enrollStatus","Use at least 6 characters for this session PIN.");return;}
 setText("enrollStatus","Capturing face...");
 try{const r=await detect();if(!r)throw Error("Exactly one clearly visible face is required. Avoid multiple faces.");
 descriptor=new Float32Array(r.descriptor);pinHash=await hashPin(pin);$("pin").value="";
 setText("enrollStatus","Driver enrolled for this tab session. Not saved to server or browser storage.");
 setText("identity","Enrolled (unverified)");
 }catch(e){setText("enrollStatus",e.message);}
}
async function verify(){
 if(!descriptor||!pinHash){setText("identity","Enroll a driver first");return;}
 try{
 const pin=$("verifyPin").value;$("verifyPin").value="";
 const pinOK=await hashPin(pin)===pinHash;
 const r=await detect();if(!r){setText("identity","Face not detected");return;}
 const d=faceapi.euclideanDistance(descriptor,r.descriptor);
 const matched=d<.47; // A demonstration threshold, not calibrated for security.
 if(pinOK&&matched){virtualIgnition=true;setText("ignition","AUTHORIZED (SIMULATION)");setText("identity","Matched + PIN accepted");}
 else{setText("identity","Not authorized");setStatus("Demo verification failed. No vehicle hardware is controlled.");}
 }catch(e){setText("identity","Verification error");setStatus(e.message);}
}
$("cameraStart").addEventListener("click",start);$("cameraStop").addEventListener("click",stop);
$("enroll").addEventListener("click",enroll);$("verify").addEventListener("click",verify);
$("resetIgnition").addEventListener("click",()=>{virtualIgnition=false;setText("ignition","LOCKED");setText("identity","Not checked");});
window.addEventListener("pagehide",()=>{stop();descriptor=null;pinHash=null;});
