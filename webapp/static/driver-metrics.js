"use strict";
/* Research-only eye/head metrics, not a validated driver safety system. */
(function(root){
 class DriverFatigueMetrics {
  constructor(){this.reset();}
  reset(){this.blinks=0;this.resetTracking();}
  resetTracking(){this.last=null;this.eyeStart=null;this.headStart=null;this.yawnStart=null;this.closedStart=null;this.samples=[];this.prevClosed=false;this.prevWarning=false;}
  update({time,eyeClosed,headAway,yawning}){
   if(!Number.isFinite(time))throw Error("Timestamp must be finite");
   if(this.last!==null&&(time<this.last||time-this.last>2000))this.resetTracking();
   const delta=this.last===null?0:Math.min(time-this.last,1000);
   // Allocate the elapsed interval only when both observations are available.
   if(this.last!==null&&delta>0){
    this.samples.push({end:time,start:time-delta,closed:this.prevClosed&&eyeClosed});
   }
   this.last=time;
   const cutoff=time-30000;
   this.samples=this.samples.filter(s=>s.end>cutoff);
   const total=this.samples.reduce((n,s)=>n+Math.max(0,s.end-Math.max(s.start,cutoff)),0);
   const closedMs=this.samples.reduce((n,s)=>n+(s.closed?Math.max(0,s.end-Math.max(s.start,cutoff)):0),0);
   const perclos=total>=10000?closedMs/total:null;
   if(eyeClosed){
    if(this.closedStart===null)this.closedStart=time;
    if(this.eyeStart===null)this.eyeStart=time;
   }else{
    const duration=this.closedStart===null?0:time-this.closedStart;
    if(this.prevClosed&&duration>=60&&duration<=800)this.blinks++;
    this.closedStart=null;this.eyeStart=null;
   }
   this.prevClosed=eyeClosed;
   this.headStart=headAway?(this.headStart===null?time:this.headStart):null;
   this.yawnStart=yawning?(this.yawnStart===null?time:this.yawnStart):null;
   const prolongedClosure=this.eyeStart!==null&&time-this.eyeStart>=1500;
   const elevatedClosure=perclos!==null&&perclos>=0.40;
   const fatigue=prolongedClosure||elevatedClosure;
   const eyeClosureMs=this.eyeStart===null?0:time-this.eyeStart;
   const distraction=this.headStart!==null&&time-this.headStart>=800;
   const prolongedYawn=this.yawnStart!==null&&time-this.yawnStart>=1200;
   return {blinks:this.blinks,perclos,eyeClosed,eyeClosureMs,fatigue,prolongedClosure,elevatedClosure,distraction,prolongedYawn,windowSeconds:total/1000};
  }
 }
 if(typeof module!=="undefined"&&module.exports)module.exports={DriverFatigueMetrics};
 root.DriverFatigueMetrics=DriverFatigueMetrics;
})(typeof window!=="undefined"?window:globalThis);
