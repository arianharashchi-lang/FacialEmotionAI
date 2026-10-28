"use strict";
const $=id=>document.getElementById(id);
const video=$("camera");
let stream=null,modelsReady=false,busy=false,timer=null,descriptor=null,pinHash=null,eyesClosedSince=null,headAwaySince=null,yawnSince=null,liveDetection=null;
let virtualIgnition=false,authFailures=0,lockedUntil=0,blinkCount=0,wasEyesClosed=false,closedFrameCount=0,soundEnabled=false,audioContext=null,lastSoundAt=0;
let previousProlongedClosure=false,previousElevatedClosure=false,previousDistraction=false,previousYawn=false;
let fatigueWarningActive=false,activeSpeechPriority=0,speechToken=0;
let eyeWasClosed=false,eyeClosedCandidateAt=null,qualifyingEyeClosures=0,eyeClosureWarningIssued=false,pullOverWarningActive=false,pullOverSpeechLastAt=0,sirenTimer=null,sirenStep=0;
let stationaryWatchId=null,stationarySince=null,stationarySamples=0,vehicleMotionState="unknown",lastReportedSpeedMps=null;
const alertHistory=[];
const eventCounts={fatigue:0,head:0,gaze:0,yawn:0,phone:0};
const sessionStart=Date.now();
const sessionAnalyticsEvents=[];
function downloadLocal(name,blob){
 const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=name;a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function renderAnalytics(){
 const report=window.DriverSessionAnalytics.summarize(sessionAnalyticsEvents,sessionStart,Date.now());
 const attention=window.DriverSessionAnalytics.calculateScore(sessionAnalyticsEvents);
 setText("analyticsTotal",String(report.total));
 setText("sessionMinutes",String(report.durationMinutes));
 setText("attentionScore",String(attention.score));
 setText("attentionScoreLabel",attention.level+" — heuristic only, not a validated safety score.");
 const chart=$("categoryChart");chart.replaceChildren();
 for(const type of window.DriverSessionAnalytics.TYPES){
  const row=document.createElement("div");row.className="report-bar";
  const label=document.createElement("span");label.textContent=window.DriverSessionAnalytics.LABELS[type];
  const track=document.createElement("div");track.className="report-track";
  const fill=document.createElement("div");fill.className="report-fill";fill.style.width=(report.total?Math.round(report.counts[type]*100/report.total):0)+"%";
  const count=document.createElement("strong");count.textContent=String(report.counts[type]);
  track.append(fill);row.append(label,track,count);chart.append(row);
 }
 const timeline=$("timeChart");timeline.replaceChildren();
 for(const bucket of report.buckets.slice(-15)){
  const bar=document.createElement("div");bar.className="minute-bar";
  bar.style.height=(Math.max(2,Math.min(100,bucket.total*14)))+"%";
  bar.title="Minute "+bucket.minute+": "+bucket.total+" warning(s)";
  bar.setAttribute("aria-label",bar.title);timeline.append(bar);
 }
}
function downloadFullReport(){
 const markup=window.DriverSessionAnalytics.reportHTML(sessionAnalyticsEvents,sessionStart,Date.now());
 downloadLocal("facialemotionai-driver-report.html",new Blob([markup],{type:"text/html;charset=utf-8"}));
}

let phoneModel=null,phoneEnabled=false,phoneBusy=false,phoneLastAt=0,phoneSeenSince=null,phoneCandidateFrames=0,phoneMissFrames=0,phoneAlertActive=false,phoneLastAlertAt=0;
const REPEATED_ALERT_MS=4000;
let yawnWarningActive=false,yawnLastAlertAt=0;
function recordEvent(type,message){
 const name=Object.prototype.hasOwnProperty.call(eventCounts,type)?type:"head";
 eventCounts[name]++;
 sessionAnalyticsEvents.push({type:name,at:Date.now(),message});
 if(sessionAnalyticsEvents.length>1000)sessionAnalyticsEvents.shift();
 renderAnalytics();
 setText("eventCount",String(Object.values(eventCounts).reduce((a,b)=>a+b,0)));
 setText("lastEvent",message);
 logEvent(message,name);
}
function downloadEvents(){
 const rows=[["local_time","event_type","description"],...alertHistory.map(e=>[e.time,e.type,e.message])];
 const safe=v=>'"'+String(v).replace(/"/g,'""').replace(/^[=+@-]/,"'")+'"';
 const csv=rows.map(row=>row.map(safe).join(",")).join("\r\n");
 const blob=new Blob([csv],{type:"text/csv;charset=utf-8"});
 const url=URL.createObjectURL(blob),a=document.createElement("a");
 a.href=url;a.download="driver-session-events.csv";a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function loadOptionalScript(src){
 return new Promise((resolve,reject)=>{
  const el=document.createElement("script");el.src=src;el.async=true;
  el.onload=resolve;el.onerror=()=>reject(Error("Could not load optional detection library"));
  document.head.appendChild(el);
 });
}
async function enablePhoneDetection(){
 if(phoneEnabled)return;
 $("phoneEnable").disabled=true;setText("phoneState","Loading model...");
 try{
  if(!window.tf)await loadOptionalScript("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js");
  if(!window.cocoSsd)await loadOptionalScript("https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js");
  if(!window.cocoSsd||!window.tf)throw Error("Object detection libraries unavailable");
  // Wait until TensorFlow.js has selected and initialized a usable backend.
  await window.tf.ready();
  if(typeof window.cocoSsd.load!=="function")throw Error("COCO-SSD model loader unavailable");
  // Try independent mirrors: the legacy Google Storage endpoint may return 403,
  // and a mirror can fail independently. COCO-SSD supports modelUrl explicitly.
  const modelUrls=[
   "https://www.gstaticcnapps.cn/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json",
   "https://libs.wware.org/@tensorflow-models/models/ssdlite_mobilenet_v2/model.json"
  ];
  let loadError=null;
  for(const modelUrl of modelUrls){
   try{
    setText("phoneState","Loading phone model...");
    phoneModel=await window.cocoSsd.load({base:"lite_mobilenet_v2",modelUrl});
    break;
   }catch(modelError){
    console.warn("Phone model mirror failed:",modelUrl,modelError);
    loadError=modelError;
    phoneModel=null;
   }
  }
  if(!phoneModel)throw Error("All phone-model sources failed. Last error: "+(loadError&&loadError.message||"unknown error"));
  phoneEnabled=true;setText("phoneState","Monitoring (experimental)");
 }catch(e){console.error("Phone detection initialization failed:",e);setText("phoneState","Unavailable: "+e.message);$("phoneEnable").disabled=false;}
}
function phoneIsInDriverZone(item,faceBox,frameWidth,frameHeight){
 // Geometry is a weak proxy for use, so only consider phones plausibly near the
 // visible driver; a phone elsewhere in the cabin/background should not alert.
 const [x,y,w,h]=item.bbox;
 if(![x,y,w,h,frameWidth,frameHeight].every(Number.isFinite)||w<=0||h<=0)return false;
 const phoneCenterX=x+w/2,phoneCenterY=y+h/2;
 const faceCenterX=faceBox.x+faceBox.width/2;
 const minWidth=Math.max(faceBox.width*.14,frameWidth*.025);
 const minHeight=Math.max(faceBox.height*.14,frameHeight*.025);
 const plausibleShape=(w/h)>=.22&&(w/h)<=2.1;
 const nearDriverHorizontally=Math.abs(phoneCenterX-faceCenterX)<=Math.max(faceBox.width*2.0,frameWidth*.18);
 const inDriverVerticalBand=phoneCenterY>=faceBox.y-faceBox.height*.20&&phoneCenterY<=Math.min(frameHeight*.97,faceBox.y+faceBox.height*5.0);
 return w>=minWidth&&h>=minHeight&&plausibleShape&&nearDriverHorizontally&&inDriverVerticalBand;
}
function resetPhoneCandidate(){
 phoneSeenSince=null;phoneCandidateFrames=0;phoneMissFrames=0;phoneAlertActive=false;phoneLastAlertAt=0;
}
async function scanPhone(time,faceResult){
 if(!phoneEnabled||!phoneModel||phoneBusy||!stream||time-phoneLastAt<1500)return;
 phoneBusy=true;phoneLastAt=time;
 try{
  // Require a visible single driver face before interpreting phone position.
  const faceBox=faceResult&&faceResult.detection&&faceResult.detection.box;
  if(!faceBox||!video.videoWidth||!video.videoHeight){
   resetPhoneCandidate();setText("phoneState","Waiting for driver face");return;
  }
  const detections=await phoneModel.detect(video,10,.65);
  const candidates=detections.filter(item=>item.class==="cell phone"&&item.score>=.70&&phoneIsInDriverZone(item,faceBox,video.videoWidth,video.videoHeight));
  if(candidates.length){
   phoneMissFrames=0;
   if(phoneSeenSince===null)phoneSeenSince=time;
   phoneCandidateFrames++;
   if(phoneCandidateFrames>=3&&time-phoneSeenSince>=2500){
    if(!phoneAlertActive){
     recordEvent("phone","Possible driver phone use (position and persistence heuristic)");
     phoneAlertActive=true;phoneLastAlertAt=0;
    }
    if(time-phoneLastAlertAt>=REPEATED_ALERT_MS){alertPhone();phoneLastAlertAt=time;}
    setText("phoneState","Possible phone use — warning");
   }else{
    setText("phoneState","Phone near driver — verifying ("+phoneCandidateFrames+"/3)");
   }
  }else{
   phoneMissFrames++;
   if(phoneMissFrames>=2){
    resetPhoneCandidate();
    const phoneVisibleElsewhere=detections.some(item=>item.class==="cell phone"&&item.score>=.70);
    setText("phoneState",phoneVisibleElsewhere?"Phone visible outside driver zone":"No likely phone use detected");
   }else{
    setText("phoneState","Checking — one missed detection tolerated");
   }
  }
 }catch(e){setText("phoneState","Detection error: "+e.message);phoneEnabled=false;$("phoneEnable").disabled=false;}
 finally{phoneBusy=false;}
}
const metrics=new window.DriverFatigueMetrics();
const gazeTracker=new window.DriverGazeTracker();
let irisMesh=null,irisReady=false,lastIrisTime=0,irisPending=false,irisWarning=false,eyeTimer=null;
let eyeSamples=0,lastFastEyeAt=null,latestHeadAway=false,latestYawning=false,headTurnState=false,headMeshLastSeenAt=null,headTurnStartedAt=null,headTurnAlerted=false;
let eyeOpenBaseline=null,eyeThreshold=null;
const lastWarningAt=Object.create(null);
function shouldWarn(kind,time,cooldownMs=15000){
 const previous=lastWarningAt[kind];
 if(previous!==undefined&&time-previous<cooldownMs)return false;
 lastWarningAt[kind]=time;return true;
}
function clearEyeCalibration(){
 eyeOpenBaseline=null;eyeThreshold=null;
 setText("eyeRatioState","—");setText("eyeThresholdState","Calibrating");
}
function calibrateEyes(ratio){
 // Collect an approximate open-eye baseline, then detect a substantial drop.
 // Avoid treating every sample as an open eye; baseline must rise gradually.
 if(eyeOpenBaseline===null)eyeOpenBaseline=ratio;
 else if(ratio>eyeOpenBaseline)eyeOpenBaseline=0.92*eyeOpenBaseline+0.08*ratio;
 else eyeOpenBaseline=0.999*eyeOpenBaseline+0.001*ratio;
 eyeThreshold=Math.max(.10,Math.min(.30,eyeOpenBaseline*.72));
 setText("eyeRatioState",ratio.toFixed(3));
 setText("eyeThresholdState",eyeThreshold.toFixed(3));
 return ratio<eyeThreshold;
}
function showEyeSignal(signal,closed){
 setText("yawnState",signal.prolongedYawn?"Possible yawn detected":latestYawning?"Mouth open — checking":"No cue");
 fatigueWarningActive=Boolean(signal.fatigue);
 setText("blinkState",String(signal.blinks));
 setText("closureDurationState",(signal.eyeClosureMs/1000).toFixed(1)+" s");
 setText("perclosState",signal.perclos===null?"Collecting data":Math.round(signal.perclos*100)+"%");
 setText("eyeState",closed?"Likely closed":"Open");
 setText("fatigueState",signal.fatigue?"Warning":"No sustained closure");
 setText("distractionState",signal.distraction?"Warning":"No prolonged head turn");
 setText("monitorState",signal.fatigue||signal.distraction?"Attention required":"Monitoring");
 // Stage eye-closure warnings: a second sustained closure says Wake up; a third,
 // or any closure lasting 1.5 seconds, escalates to a repeating pull-over alarm.
 handleDrowsinessStage(signal,closed);
 if(!irisReady&&signal.distraction&&!previousDistraction&&shouldWarn("head",performance.now(),2500)){recordEvent("head","Prolonged head turn");alertTurnAway();}
 const now=performance.now();
 if(signal.prolongedYawn){
  if(!yawnWarningActive){
   yawnWarningActive=true;
   yawnLastAlertAt=0;
   recordEvent("yawn","Possible sustained yawn");
  }
  if(now-yawnLastAlertAt>=REPEATED_ALERT_MS){
   alertYawn();
   yawnLastAlertAt=now;
  }
 }else{
  yawnWarningActive=false;
  yawnLastAlertAt=0;
 }
 previousProlongedClosure=signal.prolongedClosure;previousElevatedClosure=signal.elevatedClosure;previousDistraction=signal.distraction;previousYawn=signal.prolongedYawn;
}
function observeFastEyes(points,now){
 if(!points||points.length<468){
  if(lastFastEyeAt!==null&&now-lastFastEyeAt>900){
   metrics.resetTracking();previousProlongedClosure=previousElevatedClosure=previousDistraction=previousYawn=false;
   setText("eyeState","Unknown");setText("closureDurationState","—");setText("perclosState","Collecting data");setText("blinkState",String(metrics.blinks));
   setText("eyeSampleState","Tracking lost");lastFastEyeAt=null;
  }
  return;
 }
 // MediaPipe 468-point Face Mesh EAR, independent of identity recognition.
 const left=ear([33,160,158,133,153,144].map(i=>points[i]));
 const right=ear([362,385,387,263,373,380].map(i=>points[i]));
 const ratio=(left+right)/2;
 if(!Number.isFinite(ratio)||ratio<0||ratio>1)return;
 const closed=calibrateEyes(ratio); // Adaptive research threshold; not clinically validated.
 const signal=metrics.update({time:now,eyeClosed:closed,headAway:latestHeadAway,yawning:latestYawning});
 lastFastEyeAt=now;eyeSamples++;
 setText("eyeSampleState",String(eyeSamples));
 showEyeSignal(signal,closed);
}
async function fastEyeCycle(){
 if(!irisReady||irisPending||!stream||video.readyState<2)return;
 irisPending=true;
 try{await irisMesh.send({image:video});}
 catch(e){irisReady=false;metrics.resetTracking();setText("eyeSampleState","Iris model error");setStatus("Fast eye tracking unavailable. Slow fallback active: "+e.message);}
 finally{irisPending=false;}
}
function updateGaze(result){
 const p=result&&result.multiFaceLandmarks&&result.multiFaceLandmarks.length===1?result.multiFaceLandmarks[0]:null;
 const now=performance.now();
 // Prefer Face Mesh for head direction: it can keep tracking during turns that
 // cause the slower face-api detector to temporarily lose the face.
 if(p&&p.length>263){
  const eyeCenter=(p[33].x+p[263].x)/2;
  const eyeWidth=Math.max(.001,Math.hypot(p[33].x-p[263].x,p[33].y-p[263].y));
  const noseOffset=Math.abs((p[1].x-eyeCenter)/eyeWidth);
  if(!headTurnState&&noseOffset>.16)headTurnState=true;
  else if(headTurnState&&noseOffset<.11)headTurnState=false;
  latestHeadAway=headTurnState;
  headMeshLastSeenAt=now;
  if(latestHeadAway){
   if(headTurnStartedAt===null)headTurnStartedAt=now;
   if(!headTurnAlerted&&now-headTurnStartedAt>=10000){
    headTurnAlerted=true;
    recordEvent("head","Head turned away for 10 seconds");
    alertTurnAway();
   }
  }else{
   headTurnStartedAt=null;
   headTurnAlerted=false;
  }
  setText("headState",latestHeadAway?"Turned away":"Centered");
  setText("distractionState",headTurnStartedAt!==null&&now-headTurnStartedAt>=10000?"Warning":"No prolonged head turn");
 }else if(headMeshLastSeenAt!==null&&now-headMeshLastSeenAt>1800){
  // A sustained tracking loss is not proof that the driver is still turned away.
  headTurnStartedAt=null;
  headTurnAlerted=false;
  latestHeadAway=false;
  headTurnState=false;
  setText("headState","Unknown");
  setText("distractionState","Tracking lost");
 }
 observeFastEyes(p,now);
 if(irisReady){
  const sustainedHeadTurn=headTurnStartedAt!==null&&now-headTurnStartedAt>=10000;
  setText("distractionState",sustainedHeadTurn?"Warning":(headMeshLastSeenAt!==null&&now-headMeshLastSeenAt>1800?"Tracking lost":"No prolonged head turn"));
 }
 const output=gazeTracker.update({time:now,points:p,headAway:latestHeadAway,eyesClosed:$("eyeState").textContent==="Likely closed"});
 setText("gazeState",output.valid?(output.side==="center"?"Centered (estimate)":output.side==="left"?"Looking left (estimate)":"Looking right (estimate)"):"Unknown");
 setText("gazeAlert",output.sustainedAway?"Possible distraction":"No sustained eye shift");
 if(output.sustainedAway&&!irisWarning){recordEvent("gaze","Sustained iris shift (experimental)");alertTurnAway();}
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
const speechQueue=[];
let currentSpeech=null;
function speakAlert(phrase,priority=1,force=false){
 if(!soundEnabled&&!force)return;
 if(!("speechSynthesis" in window)||typeof window.SpeechSynthesisUtterance!=="function"){
  setStatus("Voice unavailable in this browser. No beep will play. Try a browser with speech synthesis.");
  return;
 }
 const synth=window.speechSynthesis;
 // Do not interrupt a higher-priority alert. Queue lower-priority alerts instead.
 if(currentSpeech&&currentSpeech.priority>priority){
  if(!speechQueue.some(item=>item.phrase===phrase)){
   speechQueue.push({phrase,priority,force});
   speechQueue.sort((a,b)=>b.priority-a.priority);
  }
  return;
 }
 // Avoid restarting the same alert while it is already being spoken.
 if(currentSpeech&&currentSpeech.phrase===phrase&&currentSpeech.priority===priority)return;
 playSpeech({phrase,priority,force});
}
function playSpeech(item){
 const synth=window.speechSynthesis;
 const token=++speechToken;
 try{
  if(currentSpeech||synth.speaking||synth.pending)synth.cancel();
  currentSpeech={phrase:item.phrase,priority:item.priority,token};
  activeSpeechPriority=item.priority;
  const speech=new window.SpeechSynthesisUtterance(item.phrase);
  speech.lang="en-US";speech.rate=1;speech.pitch=1;speech.volume=1;
  const voices=synth.getVoices();
  const english=voices.find(voice=>voice.lang&&voice.lang.toLowerCase().startsWith("en"));
  if(english)speech.voice=english;
  const finish=()=>{
   if(token!==speechToken)return;
   currentSpeech=null;activeSpeechPriority=0;
   if(speechQueue.length){
    const next=speechQueue.shift();
    if(soundEnabled||next.force)playSpeech(next);
   }
  };
  speech.onstart=()=>{if(token===speechToken)setStatus("Speaking: "+item.phrase);};
  speech.onend=finish;
  speech.onerror=(e)=>{if(token===speechToken){setStatus("Voice alert failed ("+(e.error||"unknown")+"). Check device voice support and volume.");finish();}};
  synth.speak(speech);
  if(synth.paused)synth.resume();
  setStatus("Voice requested: "+item.phrase+".");
 }catch(e){currentSpeech=null;activeSpeechPriority=0;setStatus("Voice unavailable: "+e.message);}
}
function alertWakeUp(force=false){
 setStatus("Fatigue warning: Wake up voice requested.");
 speakAlert("Wake up! Wake up!",2,force);
}
function alertPullOver(){speakAlert("Pull over safely! Pull over safely!",4,false);}
function playSirenPulse(){
 if(!soundEnabled||!audioContext)return;
 try{
  if(audioContext.state==="suspended")audioContext.resume();
  const osc=audioContext.createOscillator(),gain=audioContext.createGain(),now=audioContext.currentTime;
  osc.type="square";osc.frequency.setValueAtTime(sirenStep%2===0?880:1320,now);
  gain.gain.setValueAtTime(.0001,now);gain.gain.exponentialRampToValueAtTime(.12,now+.02);gain.gain.exponentialRampToValueAtTime(.0001,now+.24);
  osc.connect(gain);gain.connect(audioContext.destination);osc.start(now);osc.stop(now+.26);sirenStep++;
 }catch(e){console.warn("Siren audio unavailable:",e);}
}
function stopStationaryMonitor(){
 if(stationaryWatchId!==null&&navigator.geolocation){
  navigator.geolocation.clearWatch(stationaryWatchId);
 }
 stationaryWatchId=null;stationarySince=null;stationarySamples=0;vehicleMotionState="unknown";lastReportedSpeedMps=null;
}
function startStationaryMonitor(){
 stopStationaryMonitor();
 if(!navigator.geolocation){
  vehicleMotionState="unknown";
  setStatus("URGENT: Pull over safely. This browser has no geolocation API, so vehicle stop cannot be verified; the alarm will remain active.");
  return;
 }
 setText("monitorState","URGENT — checking motion");
 setStatus("URGENT: Pull over safely. Checking experimental GPS speed; this is not a wheel sensor and cannot certify a safe stop.");
 stationaryWatchId=navigator.geolocation.watchPosition(position=>{
  if(!pullOverWarningActive)return;
  const coords=position.coords;
  const speed=coords&&coords.speed;
  const accuracy=coords&&coords.accuracy;
  // Only use explicit GPS speed with reasonably good reported accuracy.
  // Missing speed/poor accuracy is unknown, never proof of a stop.
  if(typeof speed!=="number"||!Number.isFinite(speed)||typeof accuracy!=="number"||accuracy>20){
   vehicleMotionState="unknown";stationarySince=null;stationarySamples=0;
   setText("monitorState","URGENT — motion unknown");
   setStatus("Urgent alarm remains active: reliable GPS speed is unavailable. GPS is only an experimental proxy, not wheel-speed data.");
   return;
  }
  lastReportedSpeedMps=speed;
  if(speed<=0.5){
   if(vehicleMotionState!=="stationary")stationarySince=Date.now();
   stationarySamples++;
   vehicleMotionState="stationary";
   setText("monitorState","Vehicle appears stationary (GPS estimate)");
   if(stationarySamples>=3&&stationarySince!==null&&Date.now()-stationarySince>=8000){
    recordEvent("fatigue","GPS speed estimate remained near zero; urgent alarm acknowledged after stationary confirmation");
    clearPullOverWarning(true);
   }else{
    setStatus("GPS suggests the vehicle may be stopped; waiting for stable readings before clearing the alarm.");
   }
  }else{
   vehicleMotionState="moving";stationarySince=null;stationarySamples=0;
   setText("monitorState","URGENT — vehicle moving");
   setStatus("Vehicle appears to be moving (GPS estimate). Urgent alarm remains active until stationary readings are stable.");
  }
 },error=>{
  vehicleMotionState="unknown";stationarySince=null;stationarySamples=0;
  setText("monitorState","URGENT — motion unknown");
  setStatus("GPS motion check failed ("+error.message+"). Urgent alarm remains active; use emergency override only for an emergency or sensor/software fault.");
 },{enableHighAccuracy:true,maximumAge:0,timeout:10000});
}
function startPullOverWarning(reason){
 if(pullOverWarningActive)return;
 pullOverWarningActive=true;pullOverSpeechLastAt=performance.now();
 recordEvent("fatigue","URGENT: "+reason+" — pull over safely");
 setText("fatigueState","URGENT — pull over safely");
 setText("monitorState","URGENT WARNING");
 setStatus("URGENT: Pull over safely at the first safe location. Repeating voice and siren continue until GPS confirms stable near-zero speed or a controlled emergency override.");
 alertPullOver();playSirenPulse();
 clearInterval(sirenTimer);
 sirenTimer=setInterval(()=>{
  if(!pullOverWarningActive)return;
  playSirenPulse();
  const now=performance.now();
  if(now-pullOverSpeechLastAt>=REPEATED_ALERT_MS){alertPullOver();pullOverSpeechLastAt=now;}
 },550);
 startStationaryMonitor();
}
function clearPullOverWarning(acknowledged=false){
 pullOverWarningActive=false;clearInterval(sirenTimer);sirenTimer=null;sirenStep=0;
 stopStationaryMonitor();
 if("speechSynthesis" in window&&currentSpeech&&currentSpeech.priority>=4){
  window.speechSynthesis.cancel();currentSpeech=null;activeSpeechPriority=0;speechToken++;
 }
 if(acknowledged){
  setStatus("Stationary state was indicated by experimental GPS readings. This is not wheel-sensor verification; proceed only when it is safe.");
  setText("monitorState","Monitoring");
 }
}
function handleDrowsinessStage(signal,closed){
 const now=performance.now();
 if(closed){
  if(!eyeWasClosed){eyeClosedCandidateAt=now;eyeClosureWarningIssued=false;}
  eyeWasClosed=true;
  const duration=eyeClosedCandidateAt===null?0:now-eyeClosedCandidateAt;
  // A single continuous 7-second closure always escalates, regardless of prior count.
  if(duration>=7000){
   startPullOverWarning("Eyes closed continuously for 7 seconds");
  }else if(duration>=3000&&!eyeClosureWarningIssued){
   eyeClosureWarningIssued=true;
   if(qualifyingEyeClosures>=2){
    startPullOverWarning("Third or subsequent sustained eye closure lasting 3 seconds");
   }else{
    recordEvent("fatigue","Sustained eye closure ("+(qualifyingEyeClosures+1)+" of first two)");
    alertWakeUp();
    setStatus("Drowsiness warning: Wake up! Eyes have been closed for 3 seconds.");
   }
  }
  return;
 }
 if(eyeWasClosed){
  const duration=eyeClosedCandidateAt===null?0:now-eyeClosedCandidateAt;
  // Count only sustained closures of at least 3 seconds, never ordinary blinks.
  if(duration>=3000){
   qualifyingEyeClosures++;
   if(qualifyingEyeClosures>=3&&!pullOverWarningActive){
    startPullOverWarning("Third or subsequent sustained eye closure");
   }
  }
 }
 eyeWasClosed=false;eyeClosedCandidateAt=null;eyeClosureWarningIssued=false;
}
function alertTurnAway(){speakAlert("Turn away!",2,false);}
function alertYawn(){speakAlert("Pull over!",3,false);}
function alertPhone(){speakAlert("Put your phone away!",2,false);}
function alertSound(){alertPhone();}
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
  if(!irisReady){
   metrics.resetTracking();previousProlongedClosure=previousElevatedClosure=previousDistraction=previousYawn=false;
   setText("eyeState","Unknown");setText("perclosState","Collecting data");setText("blinkState",String(metrics.blinks));
  }
  setText("yawnState","Unknown");
  if(!irisReady||headMeshLastSeenAt===null||performance.now()-headMeshLastSeenAt>1800)setText("headState","Unknown");
  setText("monitorState","Tracking lost");
  return;
 }
 const p=result.landmarks.positions;
 const mouth=distance(p[62],p[66])/Math.max(1,distance(p[60],p[64]));
 latestYawning=mouth>.45;
 if(!irisReady){
  const eyeCenter=(p[36].x+p[45].x)/2,eyeWidth=distance(p[36],p[45]);
  const noseOffset=Math.abs((p[30].x-eyeCenter)/Math.max(1,eyeWidth));
  // Fallback to face-api landmarks only when Face Mesh is unavailable.
  if(!headTurnState&&noseOffset>.18)headTurnState=true;else if(headTurnState&&noseOffset<.12)headTurnState=false;
  latestHeadAway=headTurnState;
  setText("headState",latestHeadAway?"Turned away":"Centered");
 }
 if(!irisReady)setText("yawnState",latestYawning?"Mouth open — checking":"No cue");
 if(!irisReady){
  // Reduced-accuracy fallback when the fast mesh cannot load.
  const closed=(ear(p.slice(36,42))+ear(p.slice(42,48)))/2<.21;
  const signal=metrics.update({time:performance.now(),eyeClosed:closed,headAway:latestHeadAway,yawning:latestYawning});
  showEyeSignal(signal,closed);
  setText("eyeSampleState","Slow fallback");
 }
}
async function cycle(){
 if(busy||!stream||!modelsReady)return;
 busy=true;
 try{
  const result=await detect();
  const time=performance.now();
  if(lastInferenceAt!==null&&time-lastInferenceAt>2000){
   if(!irisReady){metrics.resetTracking();previousProlongedClosure=previousElevatedClosure=previousDistraction=previousYawn=false;}
  }
  lastInferenceAt=time;framesAnalyzed++;
  setText("sampleState",String(framesAnalyzed));
  monitoring(result);
  await scanPhone(time,result);
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
 clearInterval(timer);timer=setInterval(cycle,300);
 clearInterval(eyeTimer);eyeTimer=setInterval(fastEyeCycle,100);
 diagnostic("Camera ready — tracking starting");
 }catch(e){stop();setStatus("Could not start: "+e.message);diagnostic("Camera/model error — see status below");
 }finally{startingCamera=false;$("cameraStart").disabled=!!stream;}
}
function stop(){clearPullOverWarning(false);eyeWasClosed=false;eyeClosedCandidateAt=null;qualifyingEyeClosures=0;eyeClosureWarningIssued=false;if("speechSynthesis" in window)window.speechSynthesis.cancel();speechQueue.length=0;currentSpeech=null;activeSpeechPriority=0;speechToken++;clearEyeCalibration();clearInterval(timer);timer=null;clearInterval(eyeTimer);eyeTimer=null;lastFastEyeAt=null;eyeSamples=0;setText("eyeSampleState","0");resetGaze();resetPhoneCandidate();yawnWarningActive=false;yawnLastAlertAt=0;if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}video.srcObject=null;setText("cameraMessage","Camera off");setText("cameraDetails","—");$("cameraStart").disabled=false;resetDiagnostics();$("enroll").disabled=true;$("verify").disabled=true;monitoring(null);setText("monitorState","Inactive");}
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
$("cameraStart").addEventListener("click",start);
$("cameraStop").addEventListener("click",()=>{
 if(pullOverWarningActive){
  const confirmOverride=window.confirm("Urgent alarm is active. Stopping the camera will end this monitoring session and silence the alarm. Use only for an emergency or a sensor/software fault. Continue?");
  if(!confirmOverride)return;
  const phrase=window.prompt("Type OVERRIDE to stop monitoring despite the active alarm:");
  if(phrase!=="OVERRIDE"){setStatus("Monitoring continues and the urgent alarm remains active.");return;}
  setStatus("Monitoring stopped by emergency override. The demo cannot verify that the vehicle is safely stopped.");
 }
 stop();
});
$("enroll").addEventListener("click",enroll);$("verify").addEventListener("click",verify);
$("resetIgnition").addEventListener("click",()=>{virtualIgnition=false;setText("ignition","LOCKED");setText("identity","Not checked");});
$("audioToggle").addEventListener("click",()=>{
 if(soundEnabled&&pullOverWarningActive){
  setStatus("Urgent pull-over alarm is active. Audio cannot be disabled from this control; use the controlled emergency override only for an emergency or system fault.");
  return;
 }
 soundEnabled=!soundEnabled;
 $("audioToggle").textContent=soundEnabled?"Disable audio alerts":"Enable audio alerts";
 if(!soundEnabled&&"speechSynthesis" in window){window.speechSynthesis.cancel();speechQueue.length=0;currentSpeech=null;activeSpeechPriority=0;speechToken++;}
 if(soundEnabled){try{audioContext=audioContext||new (window.AudioContext||window.webkitAudioContext)();audioContext.resume();}catch(e){soundEnabled=false;setStatus("Audio unavailable: "+e.message);}}
});
$("testWakeVoice").addEventListener("click",()=>alertWakeUp(true));
$("resetEvents").addEventListener("click",()=>{alertHistory.length=0;sessionAnalyticsEvents.length=0;renderAnalytics();Object.keys(eventCounts).forEach(k=>eventCounts[k]=0);setText("eventCount","0");setText("lastEvent","None");metrics.reset();clearEyeCalibration();Object.keys(lastWarningAt).forEach(k=>delete lastWarningAt[k]);setText("blinkState","0");setText("closureDurationState","0.0 s");setText("perclosState","Collecting data");resetGaze();logEvent("Session counters reset");});
$("phoneEnable").addEventListener("click",enablePhoneDetection);
$("acknowledgePullOver").addEventListener("click",()=>{
 if(!pullOverWarningActive){setStatus("No active pull-over alarm.");return;}
 const phrase=window.prompt("GPS automatically clears the urgent alarm after stable near-zero speed readings. For an emergency or suspected sensor/software fault only, type OVERRIDE:");
 if(phrase!=="OVERRIDE"){
  setStatus("The urgent alarm remains active. GPS will clear it automatically only after stable, valid near-zero speed readings.");
  return;
 }
 const emergencyConfirmed=window.confirm("EMERGENCY OVERRIDE: silence the demo alarm without verified stationary status. Use only for an immediate hazard or sensor/software fault. Continue?");
 if(!emergencyConfirmed)return;
 const finalPhrase=window.prompt("To confirm emergency override, type OVERRIDE:");
 if(finalPhrase==="OVERRIDE"){
  clearPullOverWarning(false);
  setText("fatigueState","Alarm overridden — verify system");
  setText("monitorState","OVERRIDE ACTIVE");
  setStatus("Urgent alarm manually overridden. Vehicle stop was not verified. This is a demo; stop safely as soon as possible and check the system.");
 }else setStatus("Emergency override cancelled; urgent alarm remains active.");
});
$("downloadEvents").addEventListener("click",downloadEvents);
$("downloadReport").addEventListener("click",downloadFullReport);
renderAnalytics();
window.addEventListener("pagehide",()=>{stop();descriptor=null;pinHash=null;});
