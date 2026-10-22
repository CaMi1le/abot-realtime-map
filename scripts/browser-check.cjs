// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 abot-realtime-map contributors.
const {chromium} = require('playwright');
const fs = require('fs');
const path = require('path');
let runningBrowser;
(async () => {
  const url = process.argv[2] || 'http://127.0.0.1:8890';
  const output = process.argv[3] || 'artifacts/browser';
  const seconds = Number(process.argv[4] || 300);
  fs.mkdirSync(output,{recursive:true});
  const graphics = process.env.REALTIME_MAP_SOFTWARE_RENDER === '1'
    ? ['--use-angle=swiftshader','--enable-unsafe-swiftshader']
    : ['--enable-gpu','--use-angle=vulkan'];
  const browser = await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE,
    args:['--no-sandbox',...graphics]});
  runningBrowser = browser;
  const viewport={width:Number(process.env.REALTIME_MAP_WIDTH || 1280),height:Number(process.env.REALTIME_MAP_HEIGHT || 800)};
  const context = await browser.newContext({viewport,recordVideo:{dir:output,size:viewport}});
  const page = await context.newPage();
  const errors=[];
  page.on('pageerror',err=>errors.push(String(err)));
  await page.goto(url,{waitUntil:'domcontentloaded',timeout:60000});
  const started=Date.now();
  let seenFrames=false,finished=false;
  let nextSnapshot=60000;
  while(Date.now()-started < seconds*1000){
    const text = await page.locator('body').innerText();
    if(text.includes('Frame ')&&!seenFrames){
      seenFrames=true; await page.screenshot({path:path.join(output,'first-map.png')});
    }
    if(text.includes('Finished:')){finished=true;break;}
    if(Date.now()-started>=nextSnapshot){
      await page.screenshot({path:path.join(output,`progress-${Math.round(nextSnapshot/1000)}s.png`)});
      nextSnapshot+=60000;
    }
    await page.waitForTimeout(500);
  }
  await page.screenshot({path:path.join(output,'final.png')});
  const renderer=await page.evaluate(()=>{
    const c=document.querySelector('canvas'); if(!c)return null;
    const gl=c.getContext('webgl2')||c.getContext('webgl'); if(!gl)return null;
    const info=gl.getExtension('WEBGL_debug_renderer_info');
    return info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  });
  const body=await page.locator('body').innerText();
  const video=page.video();
  await context.close();
  const videoPath=await video.path();
  await browser.close();
  fs.writeFileSync(path.join(output,'browser-report.json'),JSON.stringify({url,viewport,seenFrames,finished,renderer,errors,wallSeconds:(Date.now()-started)/1000,videoPath,body},null,2));
  if(errors.length)process.exitCode=1;
})().catch(async e=>{console.error(e);if(runningBrowser)await runningBrowser.close();process.exitCode=1});
