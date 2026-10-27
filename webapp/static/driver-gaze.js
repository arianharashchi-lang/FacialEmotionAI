"use strict";
/* Experimental relative iris positioning. Not calibrated to road gaze or safety use. */
(function(root){
 function ratio(points,left,right,iris){
  if(!points||points.length<478)return null;
  const a=points[left],b=points[right];
  if(!a||!b)return null;
  const center=iris.reduce((sum,i)=>sum+(points[i]?.x??NaN),0)/iris.length;
  const span=Math.abs(b.x-a.x);
  if(!Number.isFinite(center)||!Number.isFinite(span)||span<0.008)return null;
  return (center-Math.min(a.x,b.x))/span;
 }
 function estimateGaze(points){
  // Iris indices only exist with MediaPipe refineLandmarks enabled (478 landmarks).
  const left=ratio(points,33,133,[468,469,470,471,472]);
  const right=ratio(points,362,263,[473,474,475,476,477]);
  if(left===null||right===null)return null;
  if(left<0||left>1||right<0||right>1||Math.abs(left-right)>.22)return null;
  const position=(left+right)/2;
  return {position,offset:position-.5,side:position<.34?"left":position>.66?"right":"center"};
 }
 class GazeTracker {
  constructor(){this.reset();}
  reset(){this.lastTime=null;this.awaySince=null;this.lastSide="unknown";}
  update({time,points,headAway=false,eyesClosed=false}){
   if(!Number.isFinite(time))throw Error("Invalid timestamp");
   if(this.lastTime!==null&&(time<this.lastTime||time-this.lastTime>2200))this.reset();
   this.lastTime=time;
   const gaze=(!headAway&&!eyesClosed)?estimateGaze(points):null;
   if(!gaze){this.awaySince=null;this.lastSide="unknown";return {valid:false,side:"unknown",sustainedAway:false,offset:null};}
   if(gaze.side==="center"){this.awaySince=null;}
   else if(this.awaySince===null||this.lastSide!==gaze.side){this.awaySince=time;}
   this.lastSide=gaze.side;
   return {valid:true,side:gaze.side,offset:gaze.offset,sustainedAway:this.awaySince!==null&&time-this.awaySince>=10000};
  }
 }
 if(typeof module!=="undefined"&&module.exports)module.exports={GazeTracker,estimateGaze};
 root.DriverGazeTracker=GazeTracker;
})(typeof window!=="undefined"?window:globalThis);
