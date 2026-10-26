"use strict";
/* Browser-only experimental warning analytics: no persistence or uploads. */
(function(root){
 const TYPES=["fatigue","head","gaze","yawn","phone"];
 const LABELS={fatigue:"Eye closure / fatigue",head:"Head turned",gaze:"Iris shift",yawn:"Possible yawn",phone:"Phone visible"};
 function clean(events){
  return (Array.isArray(events)?events:[]).filter(e=>e&&TYPES.includes(e.type)&&Number.isFinite(e.at))
   .map(e=>({type:e.type,at:e.at,message:String(e.message||"").slice(0,200)}))
   .sort((a,b)=>a.at-b.at);
 }
 function summarize(events,sessionStart,now){
  const valid=clean(events),counts=Object.fromEntries(TYPES.map(t=>[t,0]));
  for(const e of valid)counts[e.type]++;
  const started=Number.isFinite(sessionStart)?sessionStart:now;
  const duration=Math.max(0,now-started);
  const buckets=Array.from({length:Math.max(1,Math.ceil(duration/60000))},(_,i)=>({
   minute:i+1,counts:Object.fromEntries(TYPES.map(t=>[t,0])),total:0
  }));
  for(const e of valid){
   const idx=Math.floor((e.at-started)/60000);
   if(idx>=0&&idx<buckets.length){buckets[idx].counts[e.type]++;buckets[idx].total++;}
  }
  return {counts,total:valid.length,durationMinutes:Math.ceil(duration/60000),buckets,events:valid};
 }
 // Heuristic session indicator only; event counts are uncertain cues, not verified risk.
 const PENALTIES={fatigue:15,head:8,gaze:5,yawn:3,phone:10};
 function calculateScore(events){
  const valid=clean(events);
  const penalty=valid.reduce((sum,event)=>sum+(PENALTIES[event.type]||0),0);
  const score=Math.max(0,100-penalty);
  const level=score>=85?"Few detected cues":score>=65?"Several cues":"Frequent cues";
  return {score,level,penalty,eventCount:valid.length};
 }
 function escapeHTML(value){return String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
 function reportHTML(events,start,end){
  const data=summarize(events,start,end);
  const attention=calculateScore(data.events);
  const rows=data.events.map(e=>"<tr><td>"+escapeHTML(new Date(e.at).toLocaleString())+"</td><td>"+escapeHTML(LABELS[e.type])+"</td><td>"+escapeHTML(e.message)+"</td></tr>").join("");
  const bars=TYPES.map(t=>"<div><span>"+escapeHTML(LABELS[t])+"</span><div class='track'><div class='fill' style='width:"+(data.total?Math.round(data.counts[t]*100/data.total):0)+"%'></div></div><strong>"+data.counts[t]+"</strong></div>").join("");
  return "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Driver session report</title><style>body{font:16px system-ui;max-width:900px;margin:35px auto;padding:18px;color:#17283b}h1{color:#124f74}.track{display:inline-block;background:#dce5ec;width:min(50%,300px);height:12px;margin:0 12px}.fill{background:#2482a9;height:100%}td,th{border:1px solid #d3dce4;padding:9px;text-align:left}table{border-collapse:collapse;width:100%}tr:nth-child(even){background:#f2f6f9}p{line-height:1.55}</style><h1>FacialEmotionAI — Driver session report</h1><p>Local export only. Research prototype, not safety-certified. Flags are uncertain indicators, not verified driving behavior.</p><p>Start: "+escapeHTML(new Date(start).toLocaleString())+" | End: "+escapeHTML(new Date(end).toLocaleString())+" | Events: "+data.total+"</p><p><strong>Experimental attention indicator:</strong> "+attention.score+"/100 — "+escapeHTML(attention.level)+". This is a simple heuristic from logged warnings, not a validated safety score.</p><h2>Warning categories</h2>"+bars+"<h2>Event details</h2><table><thead><tr><th>Time</th><th>Type</th><th>Note</th></tr></thead><tbody>"+rows+"</tbody></table></html>";
 }
 if(typeof module!=="undefined"&&module.exports)module.exports={summarize,reportHTML,calculateScore,TYPES,LABELS};
 root.DriverSessionAnalytics={summarize,reportHTML,calculateScore,TYPES,LABELS};
})(typeof window!=="undefined"?window:globalThis);
