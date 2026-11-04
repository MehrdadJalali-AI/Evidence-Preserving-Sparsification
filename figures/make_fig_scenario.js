// Figure 2 (worked scenario): renders fig_scenario_source.html (the EPS Scenario Lab) in figure mode.
// Usage: node figures/make_fig_scenario.js   (requires playwright + chromium)
const {chromium}=require('playwright');
(async()=>{
 const b=await chromium.launch(); const p=await b.newPage({viewport:{width:1100,height:1200}});
 const errs=[]; p.on('pageerror',e=>errs.push(e.message));
 await p.goto('file://'+require('path').resolve(__dirname,'fig_scenario_source.html')); await p.waitForTimeout(400);
 await p.evaluate(()=>{
   document.getElementById('beta').value='0.45'; panels.L='eps'; panels.R='ppr'; setupPanels(); build(false);
   // figure-mode styling
   const st=document.createElement('style'); st.textContent=`
     *{font-family:"Liberation Sans","DejaVu Sans",sans-serif!important} body{background:#fff;font-size:17px} .row.cmp{grid-template-columns:1fr 1fr!important} header,.top,#explain,.small{display:none!important}
     .wrap{padding:0;max-width:none} .cmp{margin-top:0;gap:12px} .card{padding:10px 12px;border-radius:10px}
     .panelhead select{display:none} .ans{font-size:17px} .checks{font-size:17px} .verdict{font-size:17px}
     .legend{font-size:17px;margin-top:8px} svg{border-radius:8px}
     #strip{border:1px solid #dedad0;border-radius:10px;padding:9px 12px;margin-bottom:12px;font-size:17px}
     #strip table{border-collapse:collapse;width:100%;margin-top:5px;font-size:17px}
     #strip td,#strip th{border-bottom:1px solid #eee;padding:3px 6px;text-align:center;white-space:nowrap}
     #strip th{color:#667085;font-weight:600} #strip td:first-child,#strip th:first-child{text-align:left}
     .mname{font-weight:700;font-size:17px} .ans b{font-size:16px} h2 .n{width:26px;height:26px;font-size:16.5px}`;
   document.head.appendChild(st);
   // panel titles
   document.querySelector('#pL h2').innerHTML='<span class="n">a</span><span class="mname">EPS (whole audit units)</span>';
   document.querySelector('#pR h2').innerHTML='<span class="n">b</span><span class="mname">Personalized PageRank (relevance), same budget</span>';
   // scenario strip
   const r=state.rows, strip=document.createElement('div'); strip.id='strip';
   strip.innerHTML=`<b>Question:</b> “Which of MOFs A–F has the largest HSE06 band gap?” &nbsp;·&nbsp; <b>|N(q)|</b> = ${Object.keys(G.nodes).length} nodes, <b>|J*|</b> = ${new Set(G.units.flatMap(u=>u.nodes)).size}, <b>k</b> = ${budget()}
     <table><tr><th>MOF</th>${NAMES.map(n=>`<th>${n}</th>`).join('')}</tr>
     <tr><td>PBE gap (eV)</td>${r.map(x=>`<td>${x[0].toFixed(2)}</td>`).join('')}</tr>
     <tr><td>HSE06 gap in snapshot (eV)</td>${r.map(x=>`<td>${x[1]?x[2].toFixed(2):'none → X'}</td>`).join('')}</tr>
     <tr><td>ML prediction (HSE06)</td>${r.map(x=>`<td>${x[1]&&x[3]?x[4].toFixed(2)+' (contradicts E)':'—'}</td>`).join('')}</tr></table>`;
   document.querySelector('.wrap').prepend(strip);
 });
 await p.waitForTimeout(300);
 const h=await p.evaluate(()=>document.querySelector('.wrap').getBoundingClientRect().height+4);
 await p.pdf({path:require('path').join(__dirname,'fig_scenario.pdf'),width:'1100px',height:Math.ceil(h)+'px',printBackground:true,margin:{top:0,bottom:0,left:0,right:0}});
 await p.screenshot({path:require('path').join(__dirname,'fig_scenario.png'),fullPage:true});
 const info=await p.evaluate(()=>({L:[...document.querySelectorAll('#pL .ans')].map(e=>e.innerText),R:[...document.querySelectorAll('#pR .ans')].map(e=>e.innerText),vL:document.querySelector('#pL .verdict').innerText,vR:document.querySelector('#pR .verdict').innerText,minHtmlPx:Math.min(...[...document.querySelectorAll('.wrap *')].filter(e=>!e.closest('svg')&&e.offsetParent&&[...e.childNodes].some(n=>n.nodeType===3&&n.textContent.trim())).map(e=>parseFloat(getComputedStyle(e).fontSize)))}));
 console.log(JSON.stringify(info,null,1)); console.log('errs',errs); await b.close();
})();
