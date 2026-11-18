"""Research projects and reading state, scoped to the anonymous browser profile."""
from typing import Literal
import re
import secrets
import hashlib
import networkx as nx
from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, field_validator
from researchlens.pdf import download_pdf, supported_pdf

class ProjectInput(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    brief: str = Field(default='', max_length=6000)
    @field_validator('title')
    @classmethod
    def title_not_blank(cls, value):
        if not value.strip(): raise ValueError('Give the project a title.')
        return value.strip()

class ReadingInput(BaseModel):
    status: Literal['queued', 'reading', 'read'] = 'queued'
    page: int = Field(default=1, ge=1, le=10000)
    notes: str = Field(default='', max_length=30000)


def workspace_router():
    router = APIRouter(prefix='/api')
    def owned(request, project_id):
        p = request.app.state.store.project(request.state.profile, project_id)
        if p is None: raise HTTPException(404, 'Project not found')
        return p
    def known(request, pid):
        e = request.app.state.engine
        if pid not in e.pos: raise HTTPException(404, 'Paper not found')
        return e
    def hydrate(request, p):
        e = request.app.state.engine
        p['items'] = [dict(e.summary(e.pos[pid]), reading=request.app.state.store.reading(request.state.profile,pid)) for pid in p['paper_ids'] if pid in e.pos]
        p['paper_ids'] = [x['id'] for x in p['items']]
        return p
    @router.get('/projects')
    def projects(request: Request):
        return {'items': [hydrate(request,p) for p in request.app.state.store.projects(request.state.profile)]}
    @router.post('/projects', status_code=201)
    def create(body: ProjectInput, request: Request):
        pid = secrets.token_hex(12)
        request.app.state.store.write_project(request.state.profile,pid,body.title,body.brief)
        return hydrate(request,owned(request,pid))
    @router.get('/projects/{project_id}')
    def project(project_id: str, request: Request): return hydrate(request,owned(request,project_id))
    @router.put('/projects/{project_id}')
    def update(project_id: str, body: ProjectInput, request: Request):
        owned(request,project_id)
        request.app.state.store.write_project(request.state.profile,project_id,body.title,body.brief)
        return hydrate(request,owned(request,project_id))
    @router.delete('/projects/{project_id}')
    def delete(project_id: str, request: Request):
        owned(request,project_id)
        request.app.state.store.delete_project(request.state.profile,project_id)
        return {'deleted': True}
    @router.post('/projects/{project_id}/papers/{pid}')
    def add(project_id: str, pid: str, request: Request):
        owned(request,project_id); known(request,pid)
        request.app.state.store.project_paper(request.state.profile,project_id,pid,True)
        return {'added': True}
    @router.delete('/projects/{project_id}/papers/{pid}')
    def remove(project_id: str, pid: str, request: Request):
        owned(request,project_id)
        request.app.state.store.project_paper(request.state.profile,project_id,pid,False)
        return {'removed': True}
    @router.get('/projects/{project_id}/suggestions')
    def suggestions(project_id: str, request: Request):
        p=owned(request,project_id); e=request.app.state.engine
        # Extract only indexed vocabulary. Long prose is not parsed as Boolean query syntax.
        vocabulary=e.vectorizer.vocabulary_
        words=list(dict.fromkeys(w for w in re.findall(r'[a-z][a-z0-9_-]+',(p['title']+' '+p['brief']).lower()) if w in vocabulary))[:32]
        query=' '.join(words)[:500]
        found=e.search(query,method='bm25',limit=100+len(p['paper_ids']))['items'] if query else []
        title_key=lambda title: re.sub(r'[^a-z0-9]','',title.lower())
        seen={title_key(e.by_id[pid]['title']) for pid in p['paper_ids'] if pid in e.pos}
        items=[]
        for x in found:
            key=title_key(x['title'])
            if x['id'] in p['paper_ids'] or key in seen: continue
            seen.add(key)
            items.append(dict(x,reason='Brief terms: '+', '.join(w for w in words if w in (x['title']+' '+(x.get('abstract') or '')).lower())[:180]))
            if len(items)==10: break
        return {'items':items,'query':query,'method':'BM25 over indexed terms in title and brief'}
    @router.get('/projects/{project_id}/graph')
    def project_graph(project_id: str, request: Request, neighbors: bool=True, limit:int=Query(100,ge=1,le=200)):
        p=owned(request,project_id); e=request.app.state.engine
        seeds=[pid for pid in p['paper_ids'] if pid in e.pos]
        candidates=set(seeds)
        if neighbors:
            for pid in seeds: candidates.update(e.graph.predecessors(pid)); candidates.update(e.graph.successors(pid))
        ids=seeds[:limit]+sorted(candidates-set(seeds),key=lambda pid:(-e.pagerank[pid],pid))[:max(0,limit-len(seeds))]
        g=e.graph.subgraph(ids); layout=nx.spring_layout(g.to_undirected(),seed=42,k=.75,iterations=80)
        return {'nodes':[dict(id=pid,title=e.by_id[pid]['title'],year=e.by_id[pid].get('year'),pagerank=e.pagerank[pid],community=e.communities[pid],member=pid in seeds,x=float(layout[pid][0]),y=float(layout[pid][1])) for pid in ids],
                'edges':[{'source':a,'target':b} for a,b in g.edges], 'direction':'citing → cited','project_papers':len(seeds),'candidate_nodes':len(candidates)}
    @router.get('/reading')
    def reading_list(request: Request):
        e=request.app.state.engine
        return {'items':[dict(e.summary(e.pos[r['paper_id']]),reading=r) for r in request.app.state.store.readings(request.state.profile) if r['paper_id'] in e.pos]}
    @router.get('/papers/{pid}/passages')
    def passages(pid: str, request: Request):
        known(request,pid)
        index = request.app.state.passages
        available = index is not None and pid in index.positions
        return {'available':available,'sentences':index.choices(pid) if available else []}

    @router.get('/papers/{pid}/passage-matches')
    def passage_matches(pid: str, request: Request, sentence: int = Query(ge=0), after: str | None = Query(None,max_length=200)):
        engine=known(request,pid)
        index=request.app.state.passages
        if index is None or pid not in index.positions:
            raise HTTPException(409, 'This paper has no indexed abstract passages.')
        try:
            if after is not None and index.followups is None:
                raise HTTPException(409, "Feedback pages are unavailable for this index.")
            items=index.search(pid,sentence,after=after)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        result={'items':[dict(item,title=engine.by_id[item['id']]['title'],
                       year=engine.by_id[item['id']].get('year')) for item in items]}
        if index.followups is not None and after is None and items:
            result['can_refine']=True
        return result

    @router.get('/papers/{pid}/reading')
    def reading(pid: str, request: Request):
        known(request,pid)
        return dict(request.app.state.store.reading(request.state.profile,pid),draft_key=hashlib.sha256((request.state.profile+pid).encode()).hexdigest())
    @router.put('/papers/{pid}/reading')
    def save_reading(pid: str, body: ReadingInput, request: Request):
        known(request,pid)
        request.app.state.store.write_reading(request.state.profile,pid,body.model_dump())
        return request.app.state.store.reading(request.state.profile,pid)
    @router.get('/papers/{pid}/pdf-info')
    def pdf_info(pid: str, request: Request):
        e=known(request,pid); url=e.by_id[pid].get('pdf_url')
        return {'supported':supported_pdf(url),'url':url,'note':'PDF access depends on the publisher. Reading notes stay in this browser profile.'}
    @router.get('/papers/{pid}/pdf')
    def pdf(pid: str, request: Request):
        e=known(request,pid)
        try: content=download_pdf(e.by_id[pid].get('pdf_url'))
        except ValueError as exc: raise HTTPException(422,str(exc)) from exc
        except OSError as exc: raise HTTPException(502,str(exc)) from exc
        return Response(content,media_type='application/pdf',headers={'Content-Disposition':'inline; filename="paper.pdf"'})
    return router
