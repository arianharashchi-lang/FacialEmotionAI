"use strict";
const $=id=>document.getElementById(id);
const video=$("camera");
let stream=null,modelsReady=false,busy=false,timer=null,descriptor=null,pinHash=null,eyesClosedSince=null,headAwaySince=null,yawnSince=null,liveDetection=null;
let virtualIgnition=false,authFailures=0,lockedUntil=0,blinkCount=0,wasEyesClosed=false,closedFrameCount=0,soundEnabled=false,audioContext=null,lastSoundAt=0;
let previousFatigue=false,previousDistraction=false,previousYawn=false;
const alertHistory=[];
const eventCounts={fatigue:0,head:0,gaze:0,yawn:0,phone:0};
let phoneModel=null,phoneEnabled=false,phoneBusy=false,phoneLastAt=0,phoneSeenSince=null,phoneAlertActive=false;
function recordEvent(type,message){
 const name=Object.prototype.hasOwnProperty.call(eventCounts,type)?type:"head";
 eventCounts[name]++;setText("eventCount",String(Object.values(eventCounts).reduce((a,b)=>a+b,0)));
 setText("lastEvent",message);
 logEvent(message,name);
}
function downloadEvents(){
 const rows=[["local_time","event_type","description"],...alertHistory.map(e=>[e.time,e.type,e.message])];
 const safe=v=>'"'+String(v).replace(/"/g,'""').replace(/^[=+@-]/,"'")+'"';
 const csv=rows.map(row=>row.map(safe).join(",")).join("\\r\\n");
 const blob=new Blob([csv],{type:"text/csv;charset=utf-8"});
 const url=URL.createObjectURL(blob),a=document.createElement("a");
 a.href=url;a.download="driver-session-events.csv";a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function enablePhoneDetection(){
 if(phoneEnabled)return;
 $("phoneEnable").disabled=true;setText("phoneState","Loading model...");
 try{
  if(!window.cocoSsd||!window.tf)throw Error("Object detection libraries unavailable");
  phoneModel=await cocoSsd.load({base:"lite_mobilenet_v2"});
  phoneEnabled=true;setText("phoneState","Monitoring (experimental)");
 }catch(e){setText("phoneState","Unavailable: "+e.message);$("phoneEnable").disabled=false;}
}
async function scanPhone(time){
 if(!phoneEnabled||!phoneModel||phoneBusy||!stream||time-phoneLastAt<2600)return;
 phoneBusy=true;phoneLastAt=time;
 try{
  const detections=await phoneModel.detect(video,10,.65);
  const visible=detections.some(item=>item.class==="cell phone"&&item.score>=.70);
  if(visible){
   if(phoneSeenSince===null)phoneSeenSince=time;
   if(time-phoneSeenSince>=2400&&!phoneAlertActive){
    recordEvent("phone","Possible phone visible in camera");alertSound();phoneAlertActive=true;
   }
   setText("phoneState","Possible phone detected"+(phoneAlertActive?" — warning":""));
  }else{phoneSeenSince=null;phoneAlertActive=false;setText("phoneState","No phone detected");}
 }catch(e){setText("phoneState","Detection error: "+e.message);phoneEnabled=false;$("phoneEnable").disabled=false;}
 finally{phoneBusy=false;}
}
const metrics=new window.DriverFatigueMetrics();
const gazeTracker=new window.DriverGazeTracker();
let irisMesh=null,irisReady=false,lastIrisTime=0,irisPending=false,irisWarning=false;
function updateGaze(result){
 const p=result&&result.multiFaceLandmarks&&result.multiFaceLandmarks.length===1?result.multiFaceLandmarks[0]:null;
 const output=gazeTracker.update({time:performance.now(),points:p,headAway:$("headState").textContent==="Turned away",eyesClosed:$("eyeState").textContent==="Likely closed"});
 setText("gazeState",output.valid?(output.side==="center"?"Centered (estimate)":output.side==="left"?"Looking left (estimate)":"Looking right (estimate)"):"Unknown");
 setText("gazeAlert",output.sustainedAway?"Possible distraction":"No sustained eye shift");
 if(output.sustainedAway&&!irisWarning){recordEvent("gaze","Sustained iris shift (experimental)");alertSound();}
 irisWarning=output.sustainedAway;
 if(!output.valid)irisWarning=false;
}
function resetGaze(){gazeTracker.reset();irisWarning=false;lastIrisTime=0;setText("gazeState","Unknown");setText("gazeAlert","Not available");}
async function initIris(){
 if(irisReady)return;
 if(!window.FaceMesh)throw Error("MediaPipe iris library unavailable");
 const cdn="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4";
 const instance=new window.FaceMesh({locateFile:file=>cdn+"/"+file});
 instance.setOptions({maxNumFaces:1,refineLandmarks:true,minDetectionConfidence:.6,minTrackingConfidence:.6});
 instance.onResults(updateGaze);
 irisMesh=instance;irisReady=true;
}
let lastInferenceAt=null,framesAnalyzed=0,trackingErrors=0,startingCamera=false;
function diagnostic(message){setText("diagnosticState",message);}
function resetDiagnostics(){lastInferenceAt=null;framesAnalyzed=0;trackingErrors=0;setText("sampleState","0");setText("trackingErrors","0");diagnostic("Camera not started");}
function logEvent(message,type="system"){
 alertHistory.unshift({time:new Date().toLocaleTimeString(),message,type});
 if(alertHistory.length>25)alertHistory.pop();
 const list=$("eventHistory");list.replaceChildren();
 for(const item of alertHistory){const li=document.createElement("li");li.textContent=item.time+" — "+item.message;list.appendChild(li);}
 if(!alertHistory.length){const li=document.createElement("li");li.textContent="No alerts yet.";list.appendChild(li);}
}
function alertSound(){
 if(!soundEnabled||Date.now()-lastSoundAt<5000)return;
 lastSoundAt=Date.now();
 try{
  audioContext=audioContext||new (window.AudioContext||window.webkitAudioContext)();
  if(audioContext.state==="suspended")audioContext.resume();
  const osc=audioContext.createOscillator(),gain=audioContext.createGain();
  osc.type="sine";osc.frequency.value=880;gain.gain.value=.06;
  osc.connect(gain);gain.connect(audioContext.destination);
  osc.start();osc.stop(audioContext.currentTime+.22);
 }catch(e){setStatus("Audio alerts unavailable: "+e.message);}
}
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
async function detect(){
 if(!modelsReady||!stream||video.readyState<2)return null;
 // Reject ambiguous scenes rather than silently selecting one of several faces.
 const faces=await faceapi.detectAllFaces(video,opts()).withFaceLandmarks().withFaceDescriptors();
 if(faces.length>1)throw Error("Multiple faces detected. Only the driver should be visible.");
 return faces.length===1?faces[0]:null;
}
async function stableDescriptor(){
 const samples=[];
 for(let i=0;i<3;i++){
  const r=await detect();
  if(!r)throw Error("Face not detected. Look at the camera in good light.");
  samples.push(r.descriptor);
  if(i!==2)await new Promise(resolve=>setTimeout(resolve,350));
 }
 for(let i=1;i<samples.length;i++){
  if(faceapi.euclideanDistance(samples[0],samples[i])>.38)throw Error("Unstable face capture. Hold still and retry.");
 }
 const output=new Float32Array(samples[0].length);
 for(const sample of samples){for(let i=0;i<output.length;i++)output[i]+=sample[i]/samples.length;}
 return output;
}
function ear(points){return (distance(points[1],points[5])+distance(points[2],points[4]))/(2*Math.max(.001,distance(points[0],points[3])));}
function monitoring(result){
 setText("faceState",result?"Detected":"Not detected");
 if(!result){
  metrics.reset();resetGaze();
  ["eyeState","yawnState","headState","fatigueState","distractionState"].forEach(x=>setText(x,"Unknown"));
  setText("perclosState","Collecting data");
  setText("blinkState","0");
  previousFatigue=previousDistraction=previousYawn=false;
  setText("monitorState","Tracking lost");return;
 }
 const p=result.landmarks.positions;
 const e=(ear(p.slice(36,42))+ear(p.slice(42,48)))/2;
 const mouth=distance(p[62],p[66])/Math.max(1,distance(p[60],p[64]));
 const eyeCenter=(p[36].x+p[45].x)/2,eyeWidth=distance(p[36],p[45]);
 const offset=(p[30].x-eyeCenter)/Math.max(1,eyeWidth);
 const closed=e<.21,yawning=mouth>.45,away=Math.abs(offset)>.25;
 const signal=metrics.update({time:performance.now(),eyeClosed:closed,headAway:away,yawning});
 setText("blinkState",String(signal.blinks));
 setText("perclosState",signal.perclos===null?"Collecting data":Math.round(signal.perclos*100)+"%");
 if(signal.fatigue&&!previousFatigue){recordEvent("fatigue",signal.elevatedClosure?"Elevated eye closure over observed window":"Prolonged eye closure");alertSound();}
 if(signal.distraction&&!previousDistraction){recordEvent("head","Prolonged head turn");alertSound();}
 if(signal.prolongedYawn&&!previousYawn){recordEvent("yawn","Possible yawn");}
 previousFatigue=signal.fatigue;previousDistraction=signal.distraction;previousYawn=signal.prolongedYawn;
 setText("eyeState",closed?"Likely closed":"Open");
 setText("yawnState",yawning?"Mouth open":"No cue");
 setText("headState",away?"Turned away":"Centered");
 setText("fatigueState",signal.fatigue?"Warning":"No sustained closure");
 setText("distractionState",signal.distraction?"Warning":"No prolonged head turn");
 setText("monitorState",signal.fatigue||signal.distraction?"Attention required":"Monitoring");
}
async function cycle(){
 if(busy||!stream||!modelsReady)return;
 busy=true;
 try{
  const result=await detect();
  const time=performance.now();
  if(lastInferenceAt!==null&&time-lastInferenceAt>2000){
   metrics.reset();
   previousFatigue=previousDistraction=previousYawn=false;
  }
  lastInferenceAt=time;framesAnalyzed++;
  setText("sampleState",String(framesAnalyzed));
  monitoring(result);
  if(irisReady&&!irisPending&&stream&&time-lastIrisTime>=1100){
   irisPending=true;lastIrisTime=time;
   try{await irisMesh.send({image:video});}
   catch(irisError){resetGaze();setText("gazeState","Model error");setStatus("Iris tracking unavailable: "+irisError.message);irisReady=false;}
   finally{irisPending=false;}
  }
  await scanPhone(time);
  diagnostic(result?"Tracking active":"No face detected — move into view");
 }catch(e){
  trackingErrors++;setText("trackingErrors",String(trackingErrors));
  monitoring(null);diagnostic("Tracking interrupted");
  setStatus("Tracking error: "+e.message);
 }finally{busy=false;}
}
async function start(){
 if(startingCamera||stream)return;
 startingCamera=true;
 $("cameraStart").disabled=true;
 try{
 if(!secureContext())throw Error("Camera API unavailable. Use HTTPS/localhost and a supported browser.");
 setStatus("Requesting camera...");
 stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:"user"}});
 const track=stream.getVideoTracks()[0];
 if(track){track.addEventListener("ended",()=>{stop();setStatus("Camera disconnected or permission revoked.");diagnostic("Camera disconnected");});
  const settings=track.getSettings();setText("cameraDetails",[settings.width,settings.height].every(Boolean)?settings.width+" × "+settings.height:"Camera active");}
 video.srcObject=stream;await video.play();
 setText("cameraMessage","Camera on");setStatus("Loading face models... first run requires a network connection.");
 await loadModels();
 try{await initIris();setText("gazeModelState","Iris model initialized");}
 catch(irisError){setText("gazeModelState","Unavailable");setStatus("Iris tracking unavailable; basic driver monitoring still works. "+irisError.message);}
 setStatus("Face monitoring ready. Iris detection is experimental and depends on an external model.");
 $("enroll").disabled=false;$("verify").disabled=false;
 clearInterval(timer);timer=setInterval(cycle,650);diagnostic("Camera ready — tracking starting");
 }catch(e){stop();setStatus("Could not start: "+e.message);diagnostic("Camera/model error — see status below");
 }finally{startingCamera=false;$("cameraStart").disabled=!!stream;}
}
function stop(){clearInterval(timer);timer=null;resetGaze();phoneSeenSince=null;phoneAlertActive=false;if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}video.srcObject=null;setText("cameraMessage","Camera off");setText("cameraDetails","—");$("cameraStart").disabled=false;resetDiagnostics();$("enroll").disabled=true;$("verify").disabled=true;monitoring(null);setText("monitorState","Inactive");}
async function enroll(){
 const pin=$("pin").value;
 if(!stream){setText("enrollStatus","Start camera first.");return;}
 if(pin.length<6){setText("enrollStatus","Use at least 6 characters for this session PIN.");return;}
 setText("enrollStatus","Capturing face...");
 try{if(!stream)throw Error("Start camera first.");
 const candidate=await stableDescriptor();const newHash=await hashPin(pin);descriptor=candidate;pinHash=newHash;authFailures=0;lockedUntil=0;virtualIgnition=false;setText("ignition","LOCKED");$("pin").value="";
 setText("enrollStatus","Driver enrolled for this tab session. Not saved to server or browser storage.");
 setText("identity","Enrolled (unverified)");
 }catch(e){setText("enrollStatus",e.message);}
}
async function verify(){
 if(!descriptor||!pinHash){setText("identity","Enroll a driver first");return;}
 try{
 if(Date.now()<lockedUntil){setText("identity","Temporarily locked. Try again later.");return;}
 const pin=$("verifyPin").value;$("verifyPin").value="";
 const pinOK=await hashPin(pin)===pinHash;
 const observed=await stableDescriptor();
 const d=faceapi.euclideanDistance(descriptor,observed);
 const matched=d<.47; // A demonstration threshold, not calibrated for security.
 if(pinOK&&matched){authFailures=0;virtualIgnition=true;setText("ignition","AUTHORIZED (SIMULATION)");setText("identity","Matched + PIN accepted");}
 else{authFailures++;if(authFailures>=5){lockedUntil=Date.now()+30000;authFailures=0;}setText("identity","Not authorized");setStatus("Demo verification failed. After 5 failures, checks pause for 30 seconds. No vehicle hardware is controlled.");}
 }catch(e){setText("identity","Verification error");setStatus(e.message);}
}
$("cameraStart").addEventListener("click",start);$("cameraStop").addEventListener("click",stop);
$("enroll").addEventListener("click",enroll);$("verify").addEventListener("click",verify);
$("resetIgnition").addEventListener("click",()=>{virtualIgnition=false;setText("ignition","LOCKED");setText("identity","Not checked");});
$("audioToggle").addEventListener("click",()=>{
 soundEnabled=!soundEnabled;
 $("audioToggle").textContent=soundEnabled?"Disable sound alerts":"Enable sound alerts";
 if(soundEnabled){try{audioContext=audioContext||new (window.AudioContext||window.webkitAudioContext)();audioContext.resume();}catch(e){soundEnabled=false;setStatus("Audio unavailable: "+e.message);}}
});
$("resetEvents").addEventListener("click",()=>{alertHistory.length=0;Object.keys(eventCounts).forEach(k=>eventCounts[k]=0);setText("eventCount","0");setText("lastEvent","None");metrics.reset();setText("blinkState","0");setText("perclosState","Collecting data");resetGaze();logEvent("Session counters reset");});
$("phoneEnable").addEventListener("click",enablePhoneDetection);
$("downloadEvents").addEventListener("click",downloadEvents);
window.addEventListener("pagehide",()=>{stop();descriptor=null;pinHash=null;});
