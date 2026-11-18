'use strict';
function discoveryTabs(mode,q=''){
 const suffix=q?'?q='+encodeURIComponent(q):'';
 return `<div class="discovery-tabs" role="navigation" aria-label="Search sources"><a href="#search${suffix}" ${mode==='workspace'?'aria-current="page"':''}>Mining workspace</a><a href="#catalog${suffix}" ${mode==='catalog'?'aria-current="page"':''}>Conference library</a><a href="#online${suffix}" ${mode==='online'?'aria-current="page"':''}>Search online ↗</a></div>`;
}
function discoveryCard(p,source){
 return `<article class="paper-card discovery-card"><div class="paper-meta"><span class="year">${esc(p.year||'UNDATED')}</span><span>${esc(p.venue)}</span><span class="source-badge">${esc(p.source)}</span>${p.workspace_id?'<span class="open-label">IN WORKSPACE</span>':''}</div><h3>${esc(p.title)}</h3><div class="paper-authors">${esc(short((p.authors||[]).join(', '),170))||'Author metadata unavailable'}</div><p class="abstract">${esc(p.abstract)||'Abstract unavailable in this source. Open the original paper for the full text.'}</p><div class="paper-bottom">${p.conference?`<span class="tag">${esc(p.conference)}</span>`:''}<span>${p.source==='OpenAlex'?num(p.citations)+' external citations':'Citation count unavailable'}</span>${p.workspace_id?`<button class="secondary" data-read="${esc(p.workspace_id)}">Read in workspace ↗</button><button class="text-link" data-project-add="${esc(p.workspace_id)}">Add to project +</button>`:`<button class="secondary" data-import-paper="${esc(p.id)}" data-provider="${source}">Import & read ↗</button>`}<a class="text-link" href="${esc(safeURL(p.official_url||p.url))}" target="_blank" rel="noopener">Original source ↗</a></div></article>`;
}
async function renderDiscovery(params,epoch,mode){
 const online=mode==='online',q=params.get('q')||'',conference=params.get('conference')||'';
 const info=await api('/discovery/status');if(epoch!==state.epoch)return;
 const years=Array.from({length:info.years[1]-info.years[0]+1},(_,i)=>info.years[0]+i);
 const matrix=Object.fromEntries(info.coverage.map(c=>[c.conference+'-'+c.year,c.papers]));
 $('#main').innerHTML=heading('SEARCH SOURCES',online?'Search online':'Conference library',online?'Find papers beyond the local collection. Preview public metadata, then import the papers you want to study.':`Browse ${num(info.papers)} indexed conference records from ${info.years[0]}–${info.years[1]}. Search titles, abstracts when available, and authors.`)+discoveryTabs(mode,q)+
 `<form id="discovery-form" class="search-form"><span class="search-mark" aria-hidden="true">⌕</span><input name="q" id="discovery-query" value="${esc(q)}" maxlength="500" ${online?'minlength="2" required':''} placeholder="${online?'Find a paper by title or research terms…':'Search across the conference library…'}" aria-label="${online?'Search online papers':'Search conference papers'}"><button type="submit" class="primary">${online?'Search online ↗':'Search library'}</button></form><div class="discovery-filters">${!online?`<label>Conference<select id="conference-filter"><option value="">All conferences</option>${info.conferences.map(c=>`<option ${c===conference?'selected':''}>${esc(c)}</option>`).join('')}</select></label>`:''}<label>From year<input id="discovery-from" type="number" min="1800" max="2100" placeholder="Any" value="${esc(params.get('year_min')||'')}"></label><label>To year<input id="discovery-to" type="number" min="1800" max="2100" placeholder="Any" value="${esc(params.get('year_max')||'')}"></label><span class="muted">${online?'Provider: OpenAlex · live metadata · 5-minute cache':'Local inverted index · BM25 · works offline'}</span></div>
 ${online?'<p class="discovery-note">Imports join the shared mining workspace; your projects, notes and saved lists remain private to this browser profile. Citation links appear where the source supplies references to other indexed papers. Online search requires internet access.</p>':`<details class="coverage-panel"><summary>Coverage by conference and year <span>${info.collected_at?'Snapshot '+esc(info.collected_at.slice(0,10)):'No crawl snapshot yet'}</span></summary><div class="coverage-scroll"><table><caption>Records collected, after normalized-title deduplication within each conference/year</caption><thead><tr><th>Conference</th>${years.map(y=>`<th>${y}</th>`).join('')}</tr></thead><tbody>${info.conferences.map(c=>`<tr><th>${esc(c)}</th>${years.map(y=>`<td>${num(matrix[c+'-'+y]||0)}</td>`).join('')}</tr>`).join('')}</tbody></table></div><p>Counts reflect the fetched sources, not an independently verified total of accepted papers. Missing or unpublished editions may show zero. The benchmark snapshot remains separate.</p>${info.sources.filter(s=>!['fetched','no_published_records'].includes(s.status)).length?`<ul class="source-issues">${info.sources.filter(s=>!['fetched','no_published_records'].includes(s.status)).map(s=>`<li>${esc(s.conference)} ${s.year||'2020–2026'} · ${esc(s.source)}: ${esc(s.status)}</li>`).join('')}</ul>`:''}</details>`}
 <div id="discovery-results"></div>`;
 let serial=0,offset=Math.max(0,Number(params.get('offset'))||0);offset=Math.floor(offset/20)*20;
 const form=$('#discovery-form');
 function values(){const p=new URLSearchParams({q:$('#discovery-query').value.trim(),offset:String(offset)});if(!online&&$('#conference-filter').value)p.set('conference',$('#conference-filter').value);if($('#discovery-from').value)p.set('year_min',$('#discovery-from').value);if($('#discovery-to').value)p.set('year_max',$('#discovery-to').value);return p;}
 async function search(){
  const request=++serial,host=$('#discovery-results'),v=values();
  if(online&&!v.get('q')){host.innerHTML=empty('Search beyond the current collection.','Enter a paper title or a few specific research terms. Results come from OpenAlex; they are not generated summaries.');return;}
  host.innerHTML='<div class="loading" role="status">'+(online?'Searching OpenAlex…':'Searching conference index…')+'</div>';
  try{const result=await api((online?'/online/search?':'/catalog/search?')+v);if(epoch!==state.epoch||request!==serial)return;
   history.replaceState(null,'','#'+mode+'?'+v);
   host.innerHTML=`<div class="result-head"><span><strong>${num(result.total)}</strong> results${online?' from OpenAlex':''}</span><span class="muted">${esc(result.method)} · ${Number(result.elapsed_ms).toFixed(0)} ms</span></div>`+(result.items.length?result.items.map(p=>discoveryCard(p,online?'openalex':'catalog')).join(''):empty('No matching papers found.',online?'Try a shorter title or broader research terms.':`Try another conference or year. <a class="text-link" href="#online?q=${encodeURIComponent(v.get('q'))}">Search this query online ↗</a>`))+
    `<div class="pagination"><button class="secondary" id="discovery-prev" ${offset===0?'disabled':''}>← Previous</button><span>${result.total?offset+1:0}–${Math.min(result.total,offset+20)} of ${num(result.total)}</span><button class="secondary" id="discovery-next" ${offset+20>=result.total||online&&offset>=9980?'disabled':''}>Next →</button></div>${online&&offset>=9980?'<p class="muted">Refine the query to explore beyond the first 10,000 results.</p>':''}`;
   $('#discovery-prev').onclick=()=>{offset=Math.max(0,offset-20);search();};$('#discovery-next').onclick=()=>{offset+=20;search();};
  }catch(err){if(epoch===state.epoch&&request===serial)host.innerHTML=`<div class="error" role="alert">${esc(err.message)} <button id="retry-discovery" class="secondary">Retry</button></div><a class="text-link" href="#catalog">Search the local conference library ↗</a>`;if(epoch===state.epoch&&$('#retry-discovery'))$('#retry-discovery').onclick=search;}
 }
 form.onsubmit=e=>{e.preventDefault();offset=0;search();};
 for(const id of ['conference-filter','discovery-from','discovery-to'])if($('#'+id))$('#'+id).onchange=()=>{offset=0;search();};
 await search();
}
function renderCatalog(params,epoch){return renderDiscovery(params,epoch,'catalog');}
function renderOnline(params,epoch){return renderDiscovery(params,epoch,'online');}
document.addEventListener('click',async event=>{
 const button=event.target.closest('[data-import-paper]');if(!button)return;
 const epoch=state.epoch;button.disabled=true;button.textContent='Importing…';
 try{const result=await api('/discovery/import',{method:'POST',body:JSON.stringify({id:button.dataset.importPaper,source:button.dataset.provider})});state.facets=null;state.stats=await api('/stats');
  if(result.workspace_papers)$('#corpus-count').textContent=num(result.workspace_papers)+' WORKSPACE PAPERS';
  toast(result.warning|| (result.added?'Paper imported into your workspace.':'This paper is already in your workspace.'));
  if(epoch===state.epoch)go('reading',{paper_id:result.id});
 }catch(err){toast(err.message);button.disabled=false;button.textContent='Retry import ↗';}
});
