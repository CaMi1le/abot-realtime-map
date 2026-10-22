// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 abot-realtime-map contributors.
const {chromium}=require('playwright');
const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/usr/bin/google-chrome',args:['--no-sandbox','--enable-gpu','--use-angle=vulkan']});
 const page=await browser.newPage({viewport:{width:1440,height:900}});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('http://127.0.0.1:8891');await page.waitForTimeout(4000);
 await page.screenshot({path:'artifacts/room/overview.png'});
 console.log(await page.locator('body').innerText());
 console.log('comboboxes',await page.getByRole('combobox').count());
 await page.getByRole('combobox').click();
 await page.getByRole('option',{name:'room · After loop closure',exact:true}).click();
 await page.waitForTimeout(2000);await page.screenshot({path:'artifacts/room/overview-loop.png'});
 await page.getByRole('combobox').click();
 await page.getByRole('option',{name:'walking · Raw streaming',exact:true}).click();
 await page.waitForTimeout(2000);await page.screenshot({path:'artifacts/walking/overview.png'});
 await page.getByRole('combobox').click();
 await page.getByRole('option',{name:'desk · Raw streaming',exact:true}).click();
 await page.waitForTimeout(2000);await page.screenshot({path:'artifacts/desk/overview.png'});
 console.log(JSON.stringify({errors,loopToggleTested:true,sequencesTested:3}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
