// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 abot-realtime-map contributors.
const {chromium}=require('playwright');
const fs=require('fs');
(async()=>{
 const output='artifacts/driving/viewer-check';fs.mkdirSync(output,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:'/usr/bin/google-chrome',args:['--no-sandbox','--enable-gpu','--use-angle=vulkan']});
 const page=await browser.newPage({viewport:{width:1440,height:900}});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('http://127.0.0.1:8892');await page.waitForTimeout(2500);
 const select=async(index,text)=>{await page.getByRole('combobox').nth(index).click();await page.getByRole('option',{name:text,exact:true}).click();await page.waitForTimeout(1200);};
 for(const sequence of ['kitti05-final','kitti02-final']){
  await select(0,sequence);
  await select(1,'Road chunk + neighbors');
  const slider=page.getByRole('slider');await slider.focus();await slider.press('Home');
  for(let i=0;i<12;i++){await slider.press('ArrowRight');await page.waitForTimeout(150);}
  await page.waitForTimeout(1500);
  await page.screenshot({path:`${output}/${sequence}-detail.png`});
  await select(1,'Full route overview');
  if(sequence==='kitti05-final'){await page.getByText('Show offline-aligned ground truth',{exact:true}).click();await page.waitForTimeout(800);}
  await page.screenshot({path:`${output}/${sequence}-overview.png`});
  await select(1,'Full route after offline loop');
  await page.screenshot({path:`${output}/${sequence}-loop.png`});
 }
 fs.writeFileSync(`${output}/report.json`,JSON.stringify({sequences:2,chunkSlider:true,neighbors:true,overview:true,loopToggle:true,groundTruthToggle:true,errors,body:await page.locator('body').innerText()},null,2));
 await browser.close();if(errors.length)process.exitCode=1;
})().catch(e=>{console.error(e);process.exit(1)});
