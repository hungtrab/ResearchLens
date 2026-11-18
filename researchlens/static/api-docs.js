'use strict';
const escapeDoc = s => String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
fetch('/openapi.json').then(r=>r.json()).then(schema=>{
 document.querySelector('#endpoints').innerHTML=Object.entries(schema.paths).filter(([path])=>path.startsWith('/api/')).map(([path,methods])=>Object.entries(methods).map(([method,s])=>`<section class="paper-card"><div class="paper-meta"><span class="tag">${escapeDoc(method.toUpperCase())}</span></div><h3>${escapeDoc(path)}</h3><p>${escapeDoc(s.summary)}</p>${s.parameters?.length?'<p class="muted">Parameters: '+s.parameters.map(p=>`<code>${escapeDoc(p.name)}</code> (${escapeDoc(p.in)}${p.required?', required':''})`).join(', ')+'</p>':''}${s.requestBody?'<p class="muted">JSON request body; see the linked OpenAPI schema for constraints.</p>':''}</section>`).join('')).join('');
}).catch(e=>{document.querySelector('#endpoints').textContent=e.message;});
