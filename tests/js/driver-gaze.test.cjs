"use strict";
const assert=require("node:assert/strict");
const {test}=require("node:test");
const {GazeTracker,estimateGaze}=require("../../webapp/static/driver-gaze.js");
function points(left=.5,right=.5){
 const p=Array.from({length:478},()=>({x:0,y:0}));
 p[33]={x:.1};p[133]={x:.3};p[362]={x:.7};p[263]={x:.9};
 for(let i=468;i<=472;i++)p[i]={x:.1+.2*left};
 for(let i=473;i<=477;i++)p[i]={x:.7+.2*right};
 return p;
}
test("centered iris yields center",()=>assert.equal(estimateGaze(points()).side,"center"));
test("both irises shifted to side yields side estimate",()=>assert.equal(estimateGaze(points(.85,.85)).side,"right"));
test("missing iris landmarks return unknown",()=>assert.equal(estimateGaze(points().slice(0,468)),null));
test("discordant eyes return unknown",()=>assert.equal(estimateGaze(points(.2,.85)),null));
test("sustained off-center gaze triggers only after 2 seconds of observations",()=>{
 const t=new GazeTracker();let r;
 for(const n of [0,500,1000,1500,2000])r=t.update({time:n,points:points(.8,.8)});
 assert.equal(r.sustainedAway,true);
 assert.equal(t.update({time:2300,points:points()}).sustainedAway,false);
});
test("blinking and turned head suppress eye gaze alerts",()=>{
 const t=new GazeTracker();
 t.update({time:0,points:points(.8,.8)});
 assert.equal(t.update({time:500,points:points(.8,.8),eyesClosed:true}).valid,false);
 assert.equal(t.update({time:1000,points:points(.8,.8),headAway:true}).valid,false);
 assert.equal(t.update({time:1400,points:points(.8,.8)}).sustainedAway,false);
});
test("tracking gap resets sustained gaze duration",()=>{
 const t=new GazeTracker();t.update({time:0,points:points(.8,.8)});
 assert.equal(t.update({time:3000,points:points(.8,.8)}).sustainedAway,false);
});
