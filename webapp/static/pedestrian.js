"use strict";
(()=>{
 const $=id=>document.getElementById(id);
 const video=$("camera"),canvas=$("pedestrianOverlay"),button=$("pedestrianEnable");
 if(!video||!canvas||!button)return;
 const ctx=canvas.getContext("2d");
 let session=null,enabled=false,busy=false,lastRun=0;
 const MODEL="https://cdn.jsdelivr.net/gh/arianharashchi-lang/FacialEmotionAI@2da84cd/webapp/static/models/yolov8n.onnx",SIZE=640,CONF=.45,IOU=.45;
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
   const score=data[4*n+i];if(score<CONF)continue;
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
 function draw(boxes){
  resizeCanvas();ctx.clearRect(0,0,canvas.width,canvas.height);
  for(const b of boxes){ctx.strokeStyle="#36e6a0";ctx.lineWidth=Math.max(2,canvas.width/400);ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);ctx.fillStyle="#36e6a0";ctx.font="bold 15px sans-serif";ctx.fillText("Person "+Math.round(b.score*100)+"%",b.x1,Math.max(18,b.y1-6));}
  $("pedestrianCount").textContent=String(boxes.length);
 }
 async function scan(){
  if(!enabled||busy||!session||video.readyState<2||performance.now()-lastRun<500)return;
  busy=true;lastRun=performance.now();
  try{const meta=makeInput();const out=await session.run({images:meta.tensor});const boxes=decode(out[session.outputNames[0]].data,meta);draw(boxes);status("Active — experimental");}
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
