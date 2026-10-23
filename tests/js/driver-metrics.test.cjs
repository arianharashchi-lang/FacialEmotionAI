"use strict";
const assert=require("node:assert/strict");
const {test}=require("node:test");
const {DriverFatigueMetrics}=require("../../webapp/static/driver-metrics.js");
test("long closed eyes triggers fatigue without falsely counting a blink",()=>{
 const m=new DriverFatigueMetrics();m.update({time:0,eyeClosed:true,headAway:false,yawning:false});
 const r=m.update({time:1700,eyeClosed:true,headAway:false,yawning:false});
 assert.equal(r.fatigue,true);assert.equal(r.prolongedClosure,true);
 assert.equal(m.update({time:1850,eyeClosed:false,headAway:false,yawning:false}).blinks,0);
});
test("short eye closure counts one blink",()=>{
 const m=new DriverFatigueMetrics();m.update({time:0,eyeClosed:false,headAway:false,yawning:false});
 m.update({time:100,eyeClosed:true,headAway:false,yawning:false});
 assert.equal(m.update({time:300,eyeClosed:false,headAway:false,yawning:false}).blinks,1);
 assert.equal(m.update({time:400,eyeClosed:false,headAway:false,yawning:false}).blinks,1);
});
test("missing frames reset timing, preventing false warnings",()=>{
 const m=new DriverFatigueMetrics();m.update({time:0,eyeClosed:true,headAway:true,yawning:true});
 const r=m.update({time:5000,eyeClosed:true,headAway:true,yawning:true});
 assert.equal(r.fatigue,false);assert.equal(r.distraction,false);assert.equal(r.prolongedYawn,false);
});
test("head-turn and yawn use elapsed time",()=>{
 const m=new DriverFatigueMetrics();m.update({time:0,eyeClosed:false,headAway:true,yawning:true});
 const r=m.update({time:2100,eyeClosed:false,headAway:true,yawning:true});
 // A gap over 2 s resets signals; continuous observations required.
 assert.equal(r.distraction,false);
 m.update({time:2400,eyeClosed:false,headAway:true,yawning:true});
 m.update({time:3000,eyeClosed:false,headAway:true,yawning:true});
 assert.equal(m.update({time:3900,eyeClosed:false,headAway:true,yawning:true}).prolongedYawn,true);
 assert.equal(m.update({time:4500,eyeClosed:false,headAway:true,yawning:true}).distraction,true);
});
test("PERCLOS only appears after at least 10 seconds of observed data",()=>{
 const m=new DriverFatigueMetrics();
 for(let t=0;t<=12000;t+=500)m.update({time:t,eyeClosed:t<6000,headAway:false,yawning:false});
 const r=m.update({time:12500,eyeClosed:false,headAway:false,yawning:false});
 assert.ok(r.perclos!==null);assert.ok(r.perclos>0.35&&r.perclos<0.55);
 assert.equal(r.elevatedClosure,true);
});

test("fast consecutive eye samples register a short 80 ms blink",()=>{
 const m=new DriverFatigueMetrics();
 m.update({time:0,eyeClosed:false,headAway:false,yawning:false});
 m.update({time:100,eyeClosed:true,headAway:false,yawning:false});
 const r=m.update({time:180,eyeClosed:false,headAway:false,yawning:false});
 assert.equal(r.blinks,1);
});

test("tracking gaps preserve total blink count without inventing a blink",()=>{
 const m=new DriverFatigueMetrics();
 m.update({time:0,eyeClosed:false,headAway:false,yawning:false});
 m.update({time:100,eyeClosed:true,headAway:false,yawning:false});
 assert.equal(m.update({time:230,eyeClosed:false,headAway:false,yawning:false}).blinks,1);
 assert.equal(m.update({time:4000,eyeClosed:false,headAway:false,yawning:false}).blinks,1);
 m.resetTracking();assert.equal(m.blinks,1);
 m.reset();assert.equal(m.blinks,0);
});

test("eye closure duration rises with a sustained closure, resets after opening",()=>{
 const m=new DriverFatigueMetrics();
 m.update({time:0,eyeClosed:false,headAway:false,yawning:false});
 m.update({time:100,eyeClosed:true,headAway:false,yawning:false});
 assert.equal(m.update({time:1100,eyeClosed:true,headAway:false,yawning:false}).eyeClosureMs,1000);
 assert.equal(m.update({time:1200,eyeClosed:false,headAway:false,yawning:false}).eyeClosureMs,0);
});
