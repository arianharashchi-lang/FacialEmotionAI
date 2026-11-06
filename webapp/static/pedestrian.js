"use strict";
(()=>{
 const $=id=>document.getElementById(id);
 const video=$("camera"),canvas=$("pedestrianOverlay"),button=$("pedestrianEnable");
 if(!video||!canvas||!button)return;
 const ctx=canvas.getContext("2d");
 let session=null,enabled=false,busy=false,lastRun=0,nextTrackId=1,tracks=[],zoneHits=0;
 const MODEL="https://cdn.jsdelivr.net/gh/arianharashchi-lang/FacialEmotionAI@2da84cd/webapp/static/models/yolov8n.onnx",SIZE=640,HIGH_CONF=.45,LOW_CONF=.10,IOU=.45,TRACK_IOU=.25,MAX_MISSED=8;
 function status(s){$("pedestrianState").textContent=s;}
 function resizeCanvas(){const w=video.videoWidth||640,h=video.videoHeight||480;if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}}
 function makeInput(){
  const vw=video.videoWidth,vh=video.videoHeight,scale=Math.min(SIZE/vw,SIZE/vh);
  const nw=Math.round(vw*scale),nh=Math.round(vh*scale),px=(SIZE-nw)/2,py=(SIZE-nh)/2;
  const temp=document.createElement("canvas");temp.width=SIZE;temp.height=SIZE;
  const c=temp.getContext("2d",{willReadFrequently:true});c.fillStyle="black";c.fillRect(0,0,SIZE,SIZE);c.drawImage(video,px,py,nw,nh);
  const pixels=c.getImageData(0,0,SIZE,SIZE).data,chw=new Float32Array(3*SIZE*SIZE),plane=SIZE*SIZE;
  for(let i=0;i<plane;i++){chw[i]=pixels[i*4]/255;chw[plane+i]=pixels[i*4+1]/255;chw[2*plane+i]=pixels[i*4+2]/255;}
  return {tensor:new ort.Tensor("float32",chw,[1,3,SIZE,SIZE]),scale,px,py,vw,vh};
 }
 function iou(a,b){const x1=Math.max(a.x1,b.x1),y1=Math.max(a.y1,b.y1),x2=Math.min(a.x2,b.x2),y2=Math.min(a.y2,b.y2);const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1);return inter/( (a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter+1e-6);}
 function decode(data,meta){
  const n=8400,found=[];
  for(let i=0;i<n;i++){
   const score=data[4*n+i];if(score<LOW_CONF)continue;
   const cx=data[i],cy=data[n+i],w=data[2*n+i],h=data[3*n+i];
   let x1=(cx-w/2-meta.px)/meta.scale,y1=(cy-h/2-meta.py)/meta.scale;
   let x2=(cx+w/2-meta.px)/meta.scale,y2=(cy+h/2-meta.py)/meta.scale;
   x1=Math.max(0,Math.min(meta.vw,x1));x2=Math.max(0,Math.min(meta.vw,x2));y1=Math.max(0,Math.min(meta.vh,y1));y2=Math.max(0,Math.min(meta.vh,y2));
   if(x2>x1&&y2>y1)found.push({x1,y1,x2,y2,score});
  }
  found.sort((a,b)=>b.score-a.score);const kept=[];
  for(const box of found){if(kept.every(k=>iou(k,box)<IOU))kept.push(box);if(kept.length>=30)break;}
  return kept;
 }
 function updateTracks(detections){
  const high=detections.filter(d=>d.score>=HIGH_CONF);
  const low=detections.filter(d=>d.score< HIGH_CONF);
  const matchedTracks=new Set(),matchedHigh=new Set();
  function associate(candidates){
   const pairs=[];
   for(let ti=0;ti<tracks.length;ti++){
    if(matchedTracks.has(ti))continue;
    for(let di=0;di<candidates.length;di++){
     if(candidates[di]._used)continue;
     const overlap=iou(tracks[ti].box,candidates[di]);
     if(overlap>=TRACK_IOU)pairs.push({ti,di,overlap});
    }
   }
   pairs.sort((a,b)=>b.overlap-a.overlap);
   for(const p of pairs){
    if(matchedTracks.has(p.ti)||candidates[p.di]._used)continue;
    const t=tracks[p.ti],d=candidates[p.di];
    t.box={x1:.35*t.box.x1+.65*d.x1,y1:.35*t.box.y1+.65*d.y1,x2:.35*t.box.x2+.65*d.x2,y2:.35*t.box.y2+.65*d.y2};
    t.score=d.score;t.hits++;t.missed=0;t.age++;matchedTracks.add(p.ti);candidates[p.di]._used=true;
   }
  }
  associate(high);
  associate(low);
  tracks.forEach((t,i)=>{if(!matchedTracks.has(i)){t.missed++;t.age++;}});
  tracks=tracks.filter(t=>t.missed<=MAX_MISSED);
  high.forEach(d=>{if(!d._used)tracks.push({id:nextTrackId++,box:{x1:d.x1,y1:d.y1,x2:d.x2,y2:d.y2},score:d.score,hits:1,missed:0,age:1});});
  return tracks.filter(t=>t.missed===0);
 }
 function draw(boxes){
  resizeCanvas();ctx.clearRect(0,0,canvas.width,canvas.height);
  const zx=canvas.width*.30,zy=canvas.height*.30,zw=canvas.width*.40,zh=canvas.height*.55;
  ctx.save();ctx.setLineDash([9,6]);ctx.lineWidth=3;ctx.strokeStyle="#ffbf47";ctx.strokeRect(zx,zy,zw,zh);ctx.setLineDash([]);ctx.fillStyle="#ffbf47";ctx.font="bold 14px sans-serif";ctx.fillText("TEST ZONE",zx+8,zy+20);ctx.restore();
  const inZone=boxes.some(t=>{const b=t.box,px=(b.x1+b.x2)/2,py=b.y2;return px>=zx&&px<=zx+zw&&py>=zy&&py<=zy+zh;});
  zoneHits=inZone?zoneHits+1:0;
  const zoneAlert=zoneHits>=2;
  const zoneState=$("pedestrianZoneState");
  if(zoneState)zoneState.textContent=zoneAlert?"Person in test zone":"Clear";
  for(const t of boxes){const b=t.box;ctx.strokeStyle="#36e6a0";ctx.lineWidth=Math.max(2,canvas.width/400);ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);ctx.fillStyle="#36e6a0";ctx.font="bold 15px sans-serif";ctx.fillText("ID "+t.id+" · "+Math.round(t.score*100)+"%",b.x1,Math.max(18,b.y1-6));}
  $("pedestrianCount").textContent=String(boxes.length);
 }
 async function scan(){
  if(!enabled||busy||!session||video.readyState<2||performance.now()-lastRun<500)return;
  busy=true;lastRun=performance.now();
  try{const meta=makeInput();const out=await session.run({images:meta.tensor});const detections=decode(out[session.outputNames[0]].data,meta);const boxes=updateTracks(detections);draw(boxes);status("Active — experimental");}
  catch(e){status("Inference error");console.error("Pedestrian detector:",e);enabled=false;button.disabled=false;}
  finally{busy=false;}
 }
 async function enable(){
  if(enabled)return;
  if(!window.ort){status("ONNX Runtime unavailable");return;}
  if(!video.srcObject){status("Start camera first");return;}
  button.disabled=true;status("Loading YOLO model...");
  try{
   ort.env.wasm.wasmPaths="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
   session=await ort.InferenceSession.create(MODEL,{executionProviders:["wasm"],graphOptimizationLevel:"all"});
   enabled=true;status("Model ready");button.textContent="Pedestrian detection enabled";
   setInterval(scan,150);
  }catch(e){status("Model load failed");button.disabled=false;console.error(e);}
 }
 button.addEventListener("click",enable);
 window.addEventListener("pagehide",()=>{enabled=false;if(session)session.release();});
})();
