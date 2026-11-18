'use strict';
let graphSequence=0;
/** SVG graph: actual observed citations only; bounded motion, no rendering dependency. */
function mountCitationGraph(svg, graph, width, height, interactive=true) {
 const uid='citation-arrow-'+(++graphSequence), max=Math.max(.001,...graph.nodes.map(n=>n.pagerank));
 const extent=axis=>{const values=graph.nodes.map(n=>n[axis]);return [Math.min(0,...values),Math.max(0,...values)];};
 const [xmin,xmax]=extent('x'),[ymin,ymax]=extent('y');
 const coordinate=(v,min,max,size,padding)=>max-min<.001?size/2:padding+(v-min)/(max-min)*(size-2*padding);
 const nodes=graph.nodes.map(n=>({...n,x:coordinate(n.x,xmin,xmax,width,55),y:coordinate(n.y,ymin,ymax,height,55),r:(interactive?3:2)+Math.sqrt(n.pagerank/max)*(interactive?7:4)}));
 nodes.forEach(n=>{n.initialX=n.x;n.initialY=n.y;n.vx=0;n.vy=0;n.fx=null;n.fy=null;});
 const byId=new Map(nodes.map(n=>[n.id,n]));
 const edges=graph.edges.filter(e=>byId.has(e.source)&&byId.has(e.target));
 const adjacency=new Map(nodes.map(n=>[n.id,new Set([n.id])]));
 edges.forEach(e=>{adjacency.get(e.source).add(e.target);adjacency.get(e.target).add(e.source);});
 let selected=null,hovered=null,query='',groups=false,labels=false,zoom=1,pan={x:0,y:0},drag=null,moved=false;
 svg.classList.add('obsidian-graph');svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
 if(interactive){svg.setAttribute('tabindex','0');svg.setAttribute('aria-label','Citation graph. Drag to pan, use plus and minus to zoom. Select a paper to inspect its connections.');}
 svg.innerHTML=`<defs><marker id="${uid}" viewBox="0 0 8 8" refX="15" refY="4" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#63636f"/></marker></defs><g class="graph-layer">`+
 edges.map(e=>`<line class="edge" data-source="${esc(e.source)}" data-target="${esc(e.target)}" marker-end="url(#${uid})"/>`).join('')+
 nodes.map((n,i)=>`<g class="graph-node" data-node="${esc(n.id)}"><circle class="node" r="${n.r}" ${interactive?`tabindex="0" role="button" aria-label="${esc(n.title)}"`:''}><title>${esc(n.title)}</title></circle><text class="node-label" x="${n.x>width*.6?-n.r-6:n.r+6}" text-anchor="${n.x>width*.6?'end':'start'}" y="4" data-rank="${i}">${esc(short(n.title,interactive?30:20))}</text></g>`).join('')+'</g>';
 const layer=svg.querySelector('.graph-layer'), lines=[...svg.querySelectorAll('.edge')],elements=[...svg.querySelectorAll('.graph-node')];
 let toolbar,detail;
 if(interactive){
  toolbar=document.createElement('div');toolbar.className='graph-controls';
  toolbar.innerHTML='<label class="graph-search-label"><span>Find a node</span><input type="search" placeholder="Search titles in this graph…" aria-label="Find a node"></label><label><input type="checkbox" data-groups> Groups</label><label><input type="checkbox" data-labels> Labels</label><div class="graph-zoom"><button type="button" aria-label="Zoom out">−</button><button type="button" aria-label="Reset graph view">Fit</button><button type="button" aria-label="Zoom in">+</button></div>';
  svg.before(toolbar);detail=document.createElement('div');detail.className='graph-selection';detail.setAttribute('aria-live','polite');svg.after(detail);
 }
 function position(){elements.forEach((el,i)=>{const n=nodes[i];el.setAttribute('transform',`translate(${n.x} ${n.y})`);const label=el.querySelector('text');label.setAttribute('x',n.x>width*.6?-n.r-6:n.r+6);label.setAttribute('text-anchor',n.x>width*.6?'end':'start');});lines.forEach((el,i)=>{const a=byId.get(edges[i].source),b=byId.get(edges[i].target);el.setAttribute('x1',a.x);el.setAttribute('y1',a.y);el.setAttribute('x2',b.x);el.setAttribute('y2',b.y);});}
 function transform(){layer.setAttribute('transform',`translate(${pan.x} ${pan.y}) translate(${width/2} ${height/2}) scale(${zoom}) translate(${-width/2} ${-height/2})`);}
 const colors=['#b5a1e8','#8bbfb2','#d2ad7c','#89a8ce','#ca91a3','#a9b981'];
 function paint(){
  const focus=hovered||selected,near=focus?adjacency.get(focus):null,labelBoxes=[];
  elements.forEach((el,i)=>{const n=nodes[i],match=!query||n.title.toLowerCase().includes(query),active=(!near||near.has(n.id))&&match;
   el.classList.toggle('dimmed',!active);el.classList.toggle('selected',n.id===selected);el.classList.toggle('neighbor',!!near&&near.has(n.id));
   el.querySelector('circle').style.fill=n.id===focus?'#c2afff':groups?colors[n.community%colors.length]:n.member?'#b5a1e8':'#a8a8b3';
   const box={x:n.x+(n.x>width*.6?-180:10),y:n.y};const collides=labelBoxes.some(b=>Math.abs(b.y-box.y)<17&&Math.abs(b.x-box.x)<190);const show=active&&(labels||n.id===focus||query||(i<5&&!collides));if(show)labelBoxes.push(box);el.querySelector('text').style.opacity=show?'1':'0';
  });
  lines.forEach((el,i)=>{const e=edges[i],active=focus&&(e.source===focus||e.target===focus);el.classList.toggle('highlight',!!active);el.classList.toggle('dimmed',!!focus&&!active);});
  if(detail){const n=byId.get(selected);detail.innerHTML=n?`<div><strong>${esc(n.title)}</strong><span>${n.year||'Undated'} · ${adjacency.get(n.id).size-1} neighbors in view${n.member?' · In this project':''}</span></div><button type="button" data-read="${esc(n.id)}">Read paper ↗</button>`:`<span>${nodes.length?`${nodes.length} papers · ${edges.length} observed citations · Select a node to inspect`:'Add papers to build your citation map.'}</span>`;}
 }
 position();paint();
 if(!interactive)return;
 // Link springs transmit the drag through the graph; repulsion prevents overlaps.
 // A fixed-step integrator keeps behavior consistent at 60/120 Hz. At most 200 nodes.
 const motion=matchMedia('(prefers-reduced-motion: reduce)');
 const springs=edges.map(e=>{const a=byId.get(e.source),b=byId.get(e.target);return {a,b,length:Math.max(30,Math.hypot(a.x-b.x,a.y-b.y))};});
 let frame=0,alpha=0,lastTime=0,accumulator=0;
 function stopPhysics(){if(frame)cancelAnimationFrame(frame);frame=0;alpha=0;lastTime=0;accumulator=0;nodes.forEach(n=>{n.vx=0;n.vy=0;});svg.dataset.simulating='false';}
 function step(strength){
  for(const {a,b,length} of springs){const dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy)),f=(d-length)/d*.065*strength;
   a.vx+=dx*f;a.vy+=dy*f;b.vx-=dx*f;b.vy-=dy*f;
  }
  for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
   const a=nodes[i],b=nodes[j];let dx=b.x-a.x,dy=b.y-a.y;
   if(Math.abs(dx)+Math.abs(dy)<.001){dx=(i%2?1:-1)*.1;dy=.1;}
   const d=Math.max(.1,Math.hypot(dx,dy)),gap=a.r+b.r+10;
   const repel=(90/(d*d+100)+Math.max(0,gap-d)*.12)*strength;
   a.vx-=dx/d*repel;a.vy-=dy/d*repel;b.vx+=dx/d*repel;b.vy+=dy/d*repel;
  }
  let energy=0;
  for(const n of nodes){
   if(n.fx!==null){n.x=n.fx;n.y=n.fy;n.vx=0;n.vy=0;continue;}
   // Weak centering, rather than fixed anchors, lets a connected cluster follow freely.
   n.vx=(n.vx+(width/2-n.x)*.00035*strength)*.79;
   n.vy=(n.vy+(height/2-n.y)*.00035*strength)*.79;
   const speed=Math.hypot(n.vx,n.vy);if(speed>12){n.vx*=12/speed;n.vy*=12/speed;}
   n.x+=n.vx;n.y+=n.vy;energy+=n.vx*n.vx+n.vy*n.vy;
  }
  return energy;
 }
 function animate(now){
  frame=0;
  if(!svg.isConnected||motion.matches){stopPhysics();return;}
  accumulator+=lastTime?Math.min(50,now-lastTime):1000/60;lastTime=now;
  let energy=1;
  while(accumulator>=1000/60){energy=step(alpha);alpha=drag?.id?Math.max(.55,alpha*.97):alpha*.955;accumulator-=1000/60;}
  position();
  if(drag?.id||alpha>.012&&energy>.0001){frame=requestAnimationFrame(animate);}
  else{stopPhysics();paint();}
 }
 function heat(){
  if(motion.matches){stopPhysics();for(let i=0;i<8;i++)step(.65);position();nodes.forEach(n=>{n.vx=0;n.vy=0;});return;}
  alpha=Math.max(alpha,.85);svg.dataset.simulating='true';if(!frame)frame=requestAnimationFrame(animate);
 }
 function releaseDrag(){
  if(!drag)return;const id=drag.id;
  if(id){const n=byId.get(id);n.fx=null;n.fy=null;}
  drag=null;
  if(id&&moved&&!motion.matches)heat();else if(motion.matches)stopPhysics();
 }
 toolbar.querySelector('input[type=search]').oninput=e=>{query=e.target.value.toLowerCase().trim();paint();};
 toolbar.querySelector('[data-groups]').onchange=e=>{groups=e.target.checked;paint();};toolbar.querySelector('[data-labels]').onchange=e=>{labels=e.target.checked;paint();};
 function reset(){stopPhysics();releaseDrag();stopPhysics();nodes.forEach(n=>{n.x=n.initialX;n.y=n.initialY;});position();zoom=1;pan={x:0,y:0};selected=null;query='';toolbar.querySelector('input[type=search]').value='';transform();paint();}
 const buttons=toolbar.querySelectorAll('button');buttons[0].onclick=()=>{zoom=Math.max(.35,zoom/1.25);transform();};buttons[1].onclick=reset;buttons[2].onclick=()=>{zoom=Math.min(5,zoom*1.25);transform();};
 const resetButton=svg.parentElement.querySelector('#reset-graph');if(resetButton)resetButton.onclick=reset;
 svg.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.min(5,Math.max(.35,zoom*(e.deltaY<0?1.12:1/1.12)));transform();},{passive:false});
 function point(e){const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;return p.matrixTransform(svg.getScreenCTM().inverse());}
 svg.addEventListener('pointerdown',e=>{if(e.button!==0||drag)return;const id=e.target.closest('[data-node]')?.dataset.node;const p=point(e);drag={id,start:p,last:p};moved=false;
  if(id){const n=byId.get(id);n.fx=n.x;n.fy=n.y;n.vx=0;n.vy=0;selected=id;hovered=null;paint();}else stopPhysics();
  svg.setPointerCapture(e.pointerId);
 });
 svg.addEventListener('pointermove',e=>{if(!drag){const id=e.target.closest('[data-node]')?.dataset.node||null;if(hovered!==id){hovered=id;paint();}return;}const p=point(e);if(Math.hypot(p.x-drag.start.x,p.y-drag.start.y)>4)moved=true;
  const dx=p.x-drag.last.x,dy=p.y-drag.last.y;
  if(drag.id){const n=byId.get(drag.id);n.fx+=dx/zoom;n.fy+=dy/zoom;n.x=n.fx;n.y=n.fy;position();if(moved)heat();}
  else{pan.x+=dx;pan.y+=dy;transform();}drag.last=p;
 });
 svg.addEventListener('pointerup',e=>{if(!drag)return;if(!moved){selected=drag.id||null;paint();}releaseDrag();if(svg.hasPointerCapture(e.pointerId))svg.releasePointerCapture(e.pointerId);});
 svg.addEventListener('pointercancel',releaseDrag);svg.addEventListener('lostpointercapture',releaseDrag);svg.addEventListener('pointerleave',()=>{hovered=null;paint();});
 svg.addEventListener('keydown',e=>{const id=e.target.closest('[data-node]')?.dataset.node;if(id&&(e.key==='Enter'||e.key===' ')){e.preventDefault();selected=id;paint();return;}
  if(['+','=','-','ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Escape'].includes(e.key)){e.preventDefault();if(e.key==='Escape')return reset();if(e.key==='+'||e.key==='=')zoom=Math.min(5,zoom*1.2);if(e.key==='-')zoom=Math.max(.35,zoom/1.2);if(e.key==='ArrowLeft')pan.x-=25;if(e.key==='ArrowRight')pan.x+=25;if(e.key==='ArrowUp')pan.y-=25;if(e.key==='ArrowDown')pan.y+=25;transform();}
 });
 // Entrance opacity never overrides the SVG transform used for pan and zoom.
 if(!motion.matches){const start=performance.now();function enter(now){if(!svg.isConnected||drag){layer.style.opacity='1';return;}const t=Math.min(1,(now-start)/350);layer.style.opacity=String(.3+.7*t);if(t<1)requestAnimationFrame(enter);}requestAnimationFrame(enter);}
}
