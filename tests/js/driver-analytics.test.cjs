"use strict";
const {test}=require("node:test");
const assert=require("node:assert/strict");
const {summarize,reportHTML}=require("../../webapp/static/driver-analytics.js");
test("empty session is reported without division by zero",()=>{const r=summarize([],0,0);assert.equal(r.total,0);assert.equal(r.buckets.length,1);});
test("counts and minute buckets across categories",()=>{
 const r=summarize([{type:"fatigue",at:1000},{type:"head",at:65000},{type:"phone",at:66000}],0,120000);
 assert.equal(r.total,3);assert.equal(r.counts.phone,1);assert.equal(r.buckets[0].total,1);assert.equal(r.buckets[1].total,2);
});
test("invalid and unknown categories do not contaminate report",()=>{
 const r=summarize([{type:"bad",at:10},{type:"yawn",at:NaN},{type:"gaze",at:2500}],0,60000);
 assert.equal(r.total,1);assert.equal(r.counts.gaze,1);
});
test("HTML export escapes message content",()=>{
 const out=reportHTML([{type:"head",at:1000,message:'<script>alert("x")</script>'}],0,2000);
 assert.doesNotMatch(out,/<script>/);assert.match(out,/&lt;script&gt;/);
});
