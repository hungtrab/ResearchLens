'use strict';
// Workspace views share the established API, escaping, cards, and browser profile.
let workspaceModalVersion=0;
const statusLabel={queued:'To read',reading:'Reading',read:'Read'};
function workspaceModal(title,body){
 workspaceModalVersion++;
 const dialog=document.querySelector('#workspace-dialog');
 dialog.innerHTML=`<div class="dialog-top"><h2 id="workspace-dialog-title">${esc(title)}</h2><button type="button" class="icon-button" data-close-workspace aria-label="Close dialog">✕</button></div><div class="workspace-dialog-body">${body}</div>`;
 if(!dialog.open)dialog.showModal();return dialog;
}
function formError(form,message){form.querySelector('.form-error').textContent=message;}
function projectForm(project=null){
 const dialog=workspaceModal(project?'Edit research brief':'New research project',`<form id="project-form" class="workspace-form"><label>Project title<input name="title" required maxlength="160" placeholder="e.g. Citation-aware paper recommendation" value="${esc(project?.title||'')}"></label><label>What are you trying to find out?<textarea name="brief" maxlength="6000" rows="7" placeholder="For example: I want to tell whether someone liked the battery or the camera in a phone review.">${esc(project?.brief||'')}</textarea></label><p class="muted">Your title and brief guide paper suggestions. You choose which papers belong in the project.</p><p class="form-error" role="alert"></p><div class="form-actions">${!project?'<button type="button" class="secondary" id="example-brief">Use an example brief</button>':''}<button class="primary" type="submit">${project?'Save brief':'Create project'}</button></div></form>`);
 const version=workspaceModalVersion;
 const form=dialog.querySelector('form');
 if(!project)dialog.querySelector('#example-brief').onclick=()=>{form.elements.title.value='Citation-aware paper recommendation';form.elements.brief.value='Investigate how citation networks and content similarity support academic paper recommendation. Compare personalized PageRank, BM25, and TF-IDF. Evaluate hidden-reference recovery, ranking quality, and diversity. Study graph neural networks as a possible extension.';};
 form.onsubmit=async e=>{e.preventDefault();const b=form.querySelector('[type=submit]');b.disabled=true;try{const p=await api('/projects'+(project?'/'+project.id:''),{method:project?'PUT':'POST',body:JSON.stringify({title:form.elements.title.value,brief:form.elements.brief.value})});if(version!==workspaceModalVersion){toast('Project saved.');return;}dialog.close();if(location.hash==='#projects?id='+p.id)await router();else go('projects',{id:p.id});toast(project?'Research brief updated.':'Project created. Start building your reading list.');}catch(err){formError(form,err.message);}finally{b.disabled=false;}};
}
async function addToProject(pid){
 const dialog=workspaceModal('Add paper to a project','<div class="loading">Loading your projects…</div>');
 const version=workspaceModalVersion;
 try{const data=await api('/projects');if(!dialog.open||version!==workspaceModalVersion)return;
 dialog.querySelector('.workspace-dialog-body').innerHTML=data.items.length?`<p class="muted">Choose a research project for this paper.</p><div class="project-picker">${data.items.map(p=>`<button class="project-pick" data-pick-project="${esc(p.id)}" ${p.paper_ids.includes(pid)?'disabled':''}><span><strong>${esc(p.title)}</strong><small>${p.paper_ids.length} papers</small></span><span>${p.paper_ids.includes(pid)?'Added ✓':'Add +'}</span></button>`).join('')}</div><p class="form-error" role="alert"></p>`:empty('Create your first project.','A brief keeps your papers connected to a research question.')+'<button class="primary" id="picker-new">Create project</button>';
 const create=dialog.querySelector('#picker-new');if(create)create.onclick=()=>projectForm();
 dialog.querySelectorAll('[data-pick-project]').forEach(button=>button.onclick=async()=>{button.disabled=true;try{await api('/projects/'+button.dataset.pickProject+'/papers/'+encodeURIComponent(pid),{method:'POST'});if(version===workspaceModalVersion)dialog.close();toast('Paper added to the project.');}catch(err){if(version!==workspaceModalVersion)return;dialog.querySelector('.form-error').textContent=err.message;button.disabled=false;}});
 }catch(err){if(version!==workspaceModalVersion||!dialog.open)return;dialog.querySelector('.workspace-dialog-body').innerHTML=`<div class="error">${esc(err.message)}</div>`;}
}
async function renderProjects(params,epoch){
 const id=params.get('id');
 if(!id){const {items}=await api('/projects');if(epoch!==state.epoch)return;
  $('#main').innerHTML=heading('FROM QUESTION TO EVIDENCE','Research projects','Keep a research brief, its papers, and your reading notes together.', '<button class="primary" id="new-project">New project +</button>')+
  (items.length?`<div class="project-grid">${items.map(p=>{const read=p.items.filter(x=>x.reading.status==='read').length;return `<a class="project-card" href="#projects?id=${esc(p.id)}"><div class="project-card-top"><span class="project-symbol" aria-hidden="true">⌑</span><span>${p.paper_ids.length} papers</span></div><h2>${esc(p.title)}</h2><p>${esc(short(p.brief||'Add a research brief to guide paper suggestions.',210))}</p><div class="project-progress"><div style="width:${p.items.length?100*read/p.items.length:0}%"></div></div><div class="project-card-foot"><span>${read} of ${p.items.length} read</span><span>Open project ↗</span></div></a>`;}).join('')}</div>`:`<div class="project-empty"><div class="empty-map" aria-hidden="true"><span></span><span></span><span></span><span></span><svg viewBox="0 0 240 150"><path d="M35 85L110 35L200 95L90 125L35 85M110 35L90 125"/></svg></div><h2>Give your reading a direction.</h2><p>Start with a question. Find relevant papers, trace their references, and capture what matters for your research.</p><button class="primary" id="empty-new-project">Create your first project</button></div>`);
  $('#new-project').onclick=()=>projectForm();if($('#empty-new-project'))$('#empty-new-project').onclick=()=>projectForm();return;
 }
 const p=await api('/projects/'+encodeURIComponent(id));if(epoch!==state.epoch)return;
 const read=p.items.filter(x=>x.reading.status==='read').length;
 $('#main').innerHTML=`<a class="text-link" href="#projects">← All projects</a><div class="project-heading"><div><div class="eyebrow">RESEARCH PROJECT</div><h1>${esc(p.title)}</h1></div><div class="form-actions"><button class="secondary" id="export-project">Export notes ↓</button><button class="secondary" id="edit-project">Edit brief</button></div></div><section class="brief-block"><div><span class="rail-title">RESEARCH BRIEF</span><p>${esc(p.brief)||'Add a brief to define your research question and guide paper suggestions.'}</p></div><div class="brief-progress"><strong>${read}<span> / ${p.items.length}</span></strong><span>papers read</span><div class="project-progress"><div style="width:${p.items.length?100*read/p.items.length:0}%"></div></div></div></section>
 <div class="project-layout"><section><div class="section-head"><h2>Project papers <span class="muted">/ ${p.items.length}</span></h2><a class="text-link" href="#search">Find papers +</a></div><div class="project-filter" role="group" aria-label="Filter project papers">${['all','queued','reading','read'].map(v=>`<button class="${v==='all'?'active':''}" data-status-filter="${v}" aria-pressed="${v==='all'}">${v==='all'?'All papers':statusLabel[v]}</button>`).join('')}</div><div id="project-papers"></div><div class="section-head"><h2>Related to your brief</h2></div><p class="muted">Ranked from the terms in your title and brief. Review each paper before adding it.</p><div id="project-suggestions" class="suggestion-list"><div class="loading">Matching your brief…</div></div></section><aside class="project-context"><div class="graph-wrap"><div class="graph-toolbar"><strong>Citation map</strong><label><input id="project-neighbors" type="checkbox" checked> Neighbors</label></div><div id="project-graph-host"><svg id="project-network" class="project-network" aria-label="Project citation graph"></svg></div><div class="legend"><span>Purple = project paper · gray = neighbor</span><span>Edges: citing → cited</span></div></div><p class="annotation">The map includes observed citations only. Adding a paper to a project does not create a citation.</p><button class="text-link delete-project" id="delete-project">Delete project</button></aside></div>`;
 const drawPapers=filter=>{$('#project-papers').innerHTML=p.items.filter(x=>filter==='all'||x.reading.status===filter).map(x=>`<div class="project-paper-row"><div><span class="reading-status ${x.reading.status}">${statusLabel[x.reading.status]}</span><span class="muted"> ${x.year||''}</span><h3><button class="paper-title" data-read="${esc(x.id)}">${esc(x.title)}</button></h3><p>${esc(short(x.reading.notes||x.abstract||'No abstract available.',145))}</p><div class="form-actions"><button class="text-link" data-read="${esc(x.id)}">Open reader ↗</button><button class="text-link muted" data-remove-project-paper="${esc(x.id)}">Remove</button></div></div></div>`).join('')||empty(filter==='all'?'No papers in this project yet.':'No papers with this reading status.',filter==='all'?'Add a suggestion below or search the collection.':'Switch filters or update a paper in the reader.');};drawPapers('all');
 document.querySelectorAll('[data-status-filter]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-status-filter]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});drawPapers(b.dataset.statusFilter);});
 $('#project-papers').onclick=async e=>{const b=e.target.closest('[data-remove-project-paper]');if(!b)return;b.disabled=true;try{await api('/projects/'+id+'/papers/'+b.dataset.removeProjectPaper,{method:'DELETE'});if(epoch===state.epoch)await router();}catch(err){toast(err.message);b.disabled=false;}};
 $('#edit-project').onclick=()=>projectForm(p);
 $('#export-project').onclick=()=>downloadJSON(p,'researchlens-project.json');
 $('#delete-project').onclick=()=>{const d=workspaceModal('Delete this project?',`<p><strong>${esc(p.title)}</strong> and its paper links will be removed. Your library and paper reading notes remain available.</p><div class="form-actions"><button class="secondary" data-close-workspace>Cancel</button><button class="primary" id="confirm-delete-project">Delete project</button></div><p class="form-error" role="alert"></p>`);const version=workspaceModalVersion;d.querySelector('#confirm-delete-project').onclick=async e=>{e.target.disabled=true;try{await api('/projects/'+id,{method:'DELETE'});if(version!==workspaceModalVersion){toast('Project deleted.');return;}d.close();go('projects');}catch(err){if(version!==workspaceModalVersion)return;d.querySelector('.form-error').textContent=err.message;e.target.disabled=false;}};};
 let graphSerial=0;
 async function projectGraph(){const serial=++graphSerial,host=$('#project-graph-host');host.innerHTML='<div class="loading">Drawing citation map…</div>';try{const g=await api('/projects/'+id+'/graph?neighbors='+$('#project-neighbors').checked);if(epoch!==state.epoch||serial!==graphSerial)return;host.innerHTML='<svg id="project-network" class="project-network" aria-label="Project citation graph"></svg>';drawGraph($('#project-network'),g,540,380);}catch(err){if(epoch===state.epoch)host.innerHTML=`<div class="error">${esc(err.message)}</div>`;}}
 $('#project-neighbors').onchange=projectGraph;projectGraph();
 try{const suggestions=await api('/projects/'+id+'/suggestions');if(epoch!==state.epoch)return;
  $('#project-suggestions').innerHTML=suggestions.items.length?suggestions.items.map(x=>`<article class="brief-suggestion"><span class="muted">${x.year||''} · ${esc(short(x.venue,45))}</span><h3><button class="paper-title" data-read="${esc(x.id)}">${esc(x.title)}</button></h3><p>${esc(x.reason)}</p><button class="secondary" data-attach="${esc(x.id)}">Add to project +</button></article>`).join(''):empty('No new matches for this brief.','Try adding specific methods or datasets to the brief, or discover papers manually.');
  $('#project-suggestions').onclick=async e=>{const b=e.target.closest('[data-attach]');if(!b)return;b.disabled=true;try{await api('/projects/'+id+'/papers/'+b.dataset.attach,{method:'POST'});if(epoch===state.epoch)await router();}catch(err){toast(err.message);b.disabled=false;}};
 }catch(err){if(epoch===state.epoch)$('#project-suggestions').innerHTML=`<div class="error">${esc(err.message)}</div>`;}
}
function downloadJSON(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function bindPassageFinder(panel, pid, signal) {
 const body = panel.querySelector('.passage-body');
 const endpoint = '/papers/' + encodeURIComponent(pid);
 let loaded = false;
 let requestNumber = 0;
 panel.addEventListener('toggle', async () => {
  if (!panel.open || loaded) return;
  loaded = true;
  body.innerHTML = '<p role="status">Loading this abstract…</p>';
  try {
   const {available, sentences} = await api(endpoint + '/passages', {signal});
   if (signal.aborted) return;
   if (!available) {
    body.innerHTML = '<p>This paper has no indexed abstract. You can explore its citation neighborhood below.</p>';
    return;
   }
   body.innerHTML = `<form class="passage-form"><fieldset><legend>Which idea would you like to follow?</legend><div class="passage-choices">${sentences.map((sentence, i) => `<label><input type="radio" name="sentence" value="${i}"><span>${esc(sentence)}</span></label>`).join('')}</div></fieldset><button class="secondary" type="submit" disabled>Find similar papers</button></form><div class="passage-results" aria-live="polite"></div>`;
   const form = body.querySelector('form');
   const button = form.querySelector('button');
   const results = body.querySelector('.passage-results');
   const card = (item, canRefine) => `<article><button class="paper-title" data-read="${esc(item.id)}">${esc(item.title)}</button><blockquote>${esc(item.quote)}</blockquote>${canRefine ? `<button type="button" class="text-link" data-follow="${esc(item.id)}" aria-pressed="false">Find more like this paper</button>` : ''}</article>`;
   form.onchange = () => { requestNumber++; button.disabled = false; results.replaceChildren(); };
   form.onsubmit = async event => {
    event.preventDefault();
    const selected = form.querySelector('input:checked');
    if (!selected) return;
    const version = ++requestNumber;
    button.disabled = true;
    results.textContent = 'Finding matching passages…';
    try {
     const response = await api(endpoint + '/passage-matches?sentence=' + encodeURIComponent(selected.value), {signal});
     if (signal.aborted || version !== requestNumber) return;
     results.innerHTML = response.items.length ? '<h3>Closest matches in indexed abstracts</h3>' +
      response.items.map(item => card(item,response.can_refine)).join('') +
      (response.can_refine ? '<button class="secondary" type="button" data-follow="next" aria-pressed="false">Show next three</button><div class="passage-followup"></div>' : '') : '<p>No other indexed paper was found.</p>';
    } catch (err) {
     if (!signal.aborted && version === requestNumber) results.textContent = err.message;
    } finally {
     if (!signal.aborted && version === requestNumber) button.disabled = false;
    }
   };
   results.onclick = async event => {
    const pick = event.target.closest('[data-follow]');
    if (!pick) return;
    const area = results.querySelector('.passage-followup');
    const version = ++requestNumber;
    const after = pick.dataset.follow;
    results.querySelectorAll('[data-follow]').forEach(b => b.setAttribute('aria-pressed', String(b===pick)));
    area.textContent = 'Finding three more papers…';
    const params = new URLSearchParams({sentence:form.querySelector('input:checked').value,after});
    try {
     const response = await api(endpoint + '/passage-matches?' + params, {signal});
     if (signal.aborted || version !== requestNumber) return;
     const heading = after === 'next' ? 'Next in the original list' : 'More like “' + esc(pick.closest('article').querySelector('.paper-title').textContent) + '”';
     area.innerHTML = '<h3>' + heading + '</h3>' + (response.items.map(item => card(item,false)).join('') || '<p>No further matches in this snapshot.</p>');
     area.scrollIntoView({block:'nearest'});
    } catch (err) {
     if (!signal.aborted && version===requestNumber) area.textContent=err.message;
    }
   };
  } catch (err) {
   if (!signal.aborted) { loaded = false; body.textContent = err.message + ' Close and reopen to retry.'; }
  }
 });
}
async function renderReading(params,epoch){
 const pid=params.get('paper_id');
 if(!pid){const [recent,library]=await Promise.all([api('/reading'),api('/library')]);if(epoch!==state.epoch)return;
  const seen=new Set(recent.items.map(p=>p.id));const queued=library.items.filter(p=>!seen.has(p.id));
  $('#main').innerHTML=heading('READ. CONNECT. MAKE NOTES.','Paper reading','Read the original paper, keep a page bookmark, and trace its citation neighborhood.')+
   (recent.items.length?`<div class="section-head"><h2>Continue reading</h2></div><div class="reading-grid">${recent.items.map(p=>`<a class="reading-card" href="#reading?paper_id=${esc(p.id)}"><div><span class="reading-status ${p.reading.status}">${statusLabel[p.reading.status]}</span><span class="muted">Page ${p.reading.page}</span></div><h3>${esc(p.title)}</h3><p>${esc(short(p.reading.notes||p.abstract||'Open this paper in the reader.',140))}</p><span class="text-link">Resume paper ↗</span></a>`).join('')}</div>`:'')+
   `<div class="section-head"><h2>From your library</h2><a class="text-link" href="#search">Discover papers ↗</a></div>`+(queued.length?queued.map(p=>paperCard(p)).join(''):empty('Choose a paper to start reading.','Use “Read paper” on any result, or open a paper from a project. <a class="text-link" href="#search?q=graph%20neural%20networks">Explore the collection ↗</a>'));return;
 }
 const [p,reading,info]=await Promise.all([api('/papers/'+encodeURIComponent(pid)),api('/papers/'+encodeURIComponent(pid)+'/reading'),api('/papers/'+encodeURIComponent(pid)+'/pdf-info')]);if(epoch!==state.epoch)return;
 $('#main').innerHTML=`<a class="text-link" href="#reading">← Reading desk</a><div class="reader-heading"><div class="paper-meta"><span>${p.year||''}</span><span>${esc(p.venue)}</span></div><h1>${esc(p.title)}</h1><p>${esc(short(Array.isArray(p.authors)?p.authors.join(', '):p.authors,180))}</p></div><div class="reader-toolbar"><div class="form-actions"><a class="secondary" href="${esc(safeURL(info.url||p.url))}" target="_blank" rel="noopener">Original source ↗</a><button class="secondary" data-project-add="${esc(pid)}">Add to project +</button><button class="secondary" data-save="${esc(pid)}">${state.saved.has(pid)?'Remove from library':'Save to library'}</button></div><button class="text-link" id="focus-reader" aria-pressed="false">Focus mode</button></div><details class="passage-finder" id="passage-finder"><summary>Find papers from a sentence in this abstract</summary><div class="passage-body"></div></details><div class="reader-layout"><section class="document-pane" aria-label="Paper document"><div class="document-bar"><span id="document-label">${info.supported?'Loading original PDF…':'Abstract view'}</span><span>${esc(p.source)}</span></div><div id="pdf-host" class="pdf-host"><div class="loading" role="status">${info.supported?'Opening the paper…':'Preparing abstract…'}</div></div></section><aside class="reading-context"><section class="notebook"><div class="section-head"><h2>Reading notes</h2><span id="note-status" role="status">${reading.updated?'Saved':'No notes yet'}</span></div><form id="reading-form" class="workspace-form"><div class="reading-fields"><label>Status<select name="status">${Object.entries(statusLabel).map(([key,label])=>`<option value="${key}" ${reading.status===key?'selected':''}>${label}</option>`).join('')}</select></label><label>Bookmark page<input type="number" name="page" min="1" max="10000" value="${reading.page}" required></label></div><button type="button" class="text-link" id="jump-page">Go to bookmark ↗</button><label>Your notes<textarea name="notes" maxlength="30000" placeholder="Key claim, evidence, limitations, and how this paper relates to your research…">${esc(reading.notes)}</textarea></label><p class="note-hint">Notes save automatically. Set a bookmark here to remember your page; scrolling the PDF does not update it.</p><div class="form-actions"><button class="primary" type="submit">Save notes</button><button class="text-link" type="button" id="export-reading">Export ↓</button></div><p class="form-error" role="alert"></p></form></section><div class="graph-wrap reader-map"><div class="graph-toolbar"><strong>Citation neighborhood</strong><button class="text-link" data-network="${esc(pid)}">Expand ↗</button></div><div id="reader-graph-host"><svg id="reader-network" class="reader-network" aria-label="Paper citation neighborhood"></svg></div><div class="legend">Citing → cited · select a node to read</div></div><div class="reader-related"><h3>References in this corpus</h3>${p.references.slice(0,5).map(r=>`<button class="paper-title" data-read="${esc(r.id)}">${esc(r.title)}</button>`).join('')||'<p class="muted">No local reference links observed.</p>'}</div></aside></div>`;
 let pdfURL=null,controller=new AbortController(),timer=null,dirty=false,saving=null,closed=false;
 bindPassageFinder($('#passage-finder'),pid,controller.signal);
 const form=$('#reading-form'),status=$('#note-status');
 const draftKey='researchlens-draft-'+reading.draft_key;
 try{const draft=JSON.parse(localStorage.getItem(draftKey)||'null');if(draft){form.elements.status.value=draft.status;form.elements.page.value=draft.page;form.elements.notes.value=draft.notes;dirty=true;status.textContent='Draft recovered';}}catch{}
 const payload=()=>({status:form.elements.status.value,page:Number(form.elements.page.value)||1,notes:form.elements.notes.value});
 async function save(){clearTimeout(timer);if(saving)return saving;if(!dirty)return;if(!form.checkValidity())return;
  saving=(async()=>{while(dirty){dirty=false;const body=payload();status.textContent='Saving…';try{await api('/papers/'+encodeURIComponent(pid)+'/reading',{method:'PUT',body:JSON.stringify(body)});status.textContent=dirty?'Unsaved changes':'Saved';try{if(localStorage.getItem(draftKey)===JSON.stringify(body))localStorage.removeItem(draftKey);}catch{}formError(form,'');}catch(err){dirty=true;status.textContent='Not saved';formError(form,err.message);if(closed)toast('Reading notes could not be saved. A local draft is available when you reopen this paper.');break;}}})().finally(()=>{saving=null;});return saving;
 }
 form.oninput=()=>{dirty=true;try{localStorage.setItem(draftKey,JSON.stringify(payload()));}catch{formError(form,'Local draft storage is unavailable. Save notes before leaving.');}status.textContent='Unsaved changes';clearTimeout(timer);timer=setTimeout(save,650);};form.onsubmit=e=>{e.preventDefault();dirty=true;save();};
 $('#export-reading').onclick=()=>downloadJSON({paper_id:pid,title:p.title,...payload()},'researchlens-reading-notes.json');
 $('#jump-page').onclick=()=>{if(pdfURL){const frame=$('#paper-pdf'),replacement=frame.cloneNode(false);replacement.src=pdfURL+'#page='+Math.max(1,Number(form.elements.page.value)||1)+'&navpanes=0';frame.replaceWith(replacement);}else toast('A page bookmark is available when a PDF can be opened.');};
 $('#focus-reader').onclick=e=>{const active=$('.reader-layout').classList.toggle('focus-mode');e.target.setAttribute('aria-pressed',String(active));e.target.textContent=active?'Show notes & graph':'Focus mode';};
 if(dirty)timer=setTimeout(save,650);
 window.cleanupReader=()=>{closed=true;clearTimeout(timer);save();controller.abort();if(pdfURL)URL.revokeObjectURL(pdfURL);window.cleanupReader=null;};
 function fallback(message){if(epoch!==state.epoch)return;$('#document-label').textContent='Abstract view · PDF unavailable';$('#pdf-host').innerHTML=`<div class="pdf-fallback"><div class="pdf-notice"><strong>PDF could not be displayed</strong><p>${esc(message)}</p><a class="secondary" href="${esc(safeURL(info.url||p.url))}" target="_blank" rel="noopener">Open original source ↗</a>${info.supported?'<button class="secondary" id="retry-pdf">Retry PDF</button>':''}</div><div class="abstract-document"><span class="eyebrow">METADATA ABSTRACT · NOT THE FULL PAPER</span><h2>${esc(p.title)}</h2><p>${esc(p.abstract)||'No abstract is available in this snapshot.'}</p></div></div>`;if($('#retry-pdf'))$('#retry-pdf').onclick=loadPDF;}
 async function loadPDF(){if(!info.supported)return fallback('This paper has no PDF URL from a supported public repository. You can read its abstract here or visit the publisher.');
  $('#document-label').textContent='Loading original PDF…';$('#pdf-host').innerHTML='<div class="loading" role="status">Fetching the original PDF…</div>';
  try{const response=await fetch('/api/papers/'+encodeURIComponent(pid)+'/pdf',{signal:controller.signal});if(!response.ok){let message='PDF source is unavailable.';try{message=(await response.json()).detail;}catch{}throw new Error(message);}const blob=await response.blob();if(epoch!==state.epoch)return;if(pdfURL)URL.revokeObjectURL(pdfURL);pdfURL=URL.createObjectURL(blob);$('#document-label').textContent='Original PDF';$('#pdf-host').innerHTML=`<iframe id="paper-pdf" title="Original paper PDF" src="${pdfURL}#page=${Math.max(1,Number(form.elements.page.value)||1)}&navpanes=0"></iframe><p class="pdf-mobile-help">If your browser cannot render PDFs, use “Original source” above.</p>`;}catch(err){if(err.name!=='AbortError')fallback(err.message);}
 }
 loadPDF();api('/graph?paper_id='+encodeURIComponent(pid)+'&limit=35').then(g=>{if(epoch===state.epoch)drawGraph($('#reader-network'),g,430,300);}).catch(err=>{if(epoch===state.epoch)$('#reader-graph-host').innerHTML=`<div class="error">${esc(err.message)}</div>`;});
}
document.addEventListener('click',e=>{
 const target=e.target.closest('[data-read],[data-project-add],[data-close-workspace]');if(!target)return;
 if(target.hasAttribute('data-close-workspace'))return $('#workspace-dialog').close();
 if(target.hasAttribute('data-read')){$('#paper-dialog').close();go('reading',{paper_id:target.dataset.read});}
 if(target.hasAttribute('data-project-add'))addToProject(target.dataset.projectAdd);
});
window.addEventListener('beforeunload',()=>{if(window.cleanupReader)window.cleanupReader();});
