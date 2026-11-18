"""Search a crawled catalog or live web metadata; explicitly import chosen papers."""
import json
import re
import time
from typing import Literal
from fastapi import APIRouter,HTTPException,Query,Request
from pydantic import BaseModel,Field
from researchlens.catalog import CONFERENCES,title_key
from researchlens.sources import SourceUnavailable
from researchlens.mining import MiningEngine

class ImportInput(BaseModel):
    id: str=Field(min_length=1,max_length=160,pattern=r'^[A-Za-z0-9_.-]+$')
    source: Literal['catalog','openalex']
    enrich: bool=True

def same_paper(a,b):
    if a.get('doi') and b.get('doi'):
        return a['doi'].lower().removeprefix('https://doi.org/')==b['doi'].lower().removeprefix('https://doi.org/')
    key=title_key(a['title'])
    return len(key)>=20 and key==title_key(b['title']) and a.get('year')==b.get('year')

def resolve_references(papers,aliases):
    """Resolve provider IDs without modifying the benchmark or stored source payloads."""
    return [dict(p,references=list(dict.fromkeys(aliases.get(ref,ref) for ref in p.get('references',[])))) for p in papers]

def discovery_router():
    router=APIRouter(prefix='/api')
    def decorate(request,result):
        e=request.app.state.engine
        for p in result['items']:
            imported=p['id'] if p['id'] in e.pos else request.app.state.store.imported_alias(p['id'])
            p['workspace_id']=imported if imported in e.pos else None
        return result
    @router.get('/discovery/status')
    def status(request:Request):
        state=request.app.state;result=state.catalog.stats();path=state.data_dir/'conferences/manifest.json'
        manifest=json.loads(path.read_text()) if path.exists() else {}
        result.update(conferences=CONFERENCES,years=manifest.get('years',[2020,2026]),collected_at=manifest.get('collected_at'),
                      sources=[{k:v for k,v in s.items() if k!='urls'} for s in manifest.get('sources',[])],
                      workspace_papers=len(state.engine.ids),benchmark_papers=state.benchmark_count,online_provider='OpenAlex',
                      note='Imports join the shared mining workspace; projects, notes and saved lists remain private to your browser profile. Experiments use the fixed benchmark snapshot.')
        return result
    @router.get('/catalog/search')
    def catalog_search(request:Request,q:str=Query('',max_length=500),conference:str|None=None,year_min:int|None=Query(None,ge=1800,le=2100),year_max:int|None=Query(None,ge=1800,le=2100),offset:int=Query(0,ge=0),limit:int=Query(20,ge=1,le=100)):
        if conference is not None and conference not in CONFERENCES:raise HTTPException(422,'Unknown conference')
        if year_min is not None and year_max is not None and year_min>year_max:raise HTTPException(422,'Start year must not exceed end year.')
        start=time.perf_counter();result=request.app.state.catalog.search(q,conference,year_min,year_max,offset,limit)
        return decorate(request,dict(result,elapsed_ms=round(1000*(time.perf_counter()-start),2),offset=offset))
    @router.get('/catalog/papers/{pid}')
    def catalog_paper(pid:str,request:Request):
        p=request.app.state.catalog.get(pid)
        if not p:raise HTTPException(404,'Paper not found in conference catalog')
        return decorate(request,{'items':[p]})['items'][0]
    @router.get('/online/search')
    def online(request:Request,q:str=Query(min_length=2,max_length=500),year_min:int|None=Query(None,ge=1800,le=2100),year_max:int|None=Query(None,ge=1800,le=2100),offset:int=Query(0,ge=0,le=9980,multiple_of=20)):
        if len(q.strip())<2:raise HTTPException(422,'Enter at least two non-space characters.')
        if year_min is not None and year_max is not None and year_min>year_max:raise HTTPException(422,'Start year must not exceed end year.')
        start=time.perf_counter()
        try:result=request.app.state.online.search(q.strip(),year_min,year_max,offset,20)
        except SourceUnavailable as exc:raise HTTPException(503,str(exc)) from exc
        return decorate(request,dict(result,elapsed_ms=round(1000*(time.perf_counter()-start),2),offset=offset))
    @router.post('/discovery/import')
    def import_paper(body:ImportInput,request:Request):
        state=request.app.state;alias=state.store.imported_alias(body.id)
        existing=body.id if body.id in state.engine.pos else alias
        if existing in state.engine.pos:return {'id':existing,'added':False,'warning':None}
        warning=None
        if body.source=='catalog':
            p=state.catalog.get(body.id)
            if not p:raise HTTPException(404,'Paper not found in conference catalog')
            if body.enrich and p['source']!='OpenAlex':
                try:
                    matches=state.online.search(p['title'][:500])['items']
                    match=next((x for x in matches if same_paper(p,x)),None)
                    if match:
                        original=p;p=dict(match,conference=original['conference'],venue=original['venue'],collection_topic=original['conference'],official_url=original['url'],pdf_url=original.get('pdf_url') or match.get('pdf_url'))
                        if not p.get('abstract'):p['abstract']=original.get('abstract','')
                    else:warning='Imported official metadata. No exact OpenAlex match was found for citation enrichment.'
                except SourceUnavailable:warning='Imported official metadata. Citation enrichment is currently unavailable.'
        else:
            if not re.fullmatch(r'W[0-9]+',body.id):raise HTTPException(422,'Invalid OpenAlex ID')
            try:p=state.online.paper(body.id)
            except KeyError as exc:raise HTTPException(404,'Paper not found at OpenAlex') from exc
            except SourceUnavailable as exc:raise HTTPException(503,str(exc)) from exc
        # Build a new immutable engine and swap it only after persistence succeeds.
        # Existing requests keep their captured engine; concurrent imports serialize here.
        with state.import_lock:
            e=state.engine
            duplicate=next((x for x in e.papers if x['id']==p['id'] or same_paper(x,p)),None)
            aliases=list(set([body.id,p['id']]+p.get('alternate_ids',[])))
            mapping=state.store.aliases()
            if duplicate:
                mapping.update(dict.fromkeys(aliases,duplicate['id']))
                new_engine=MiningEngine(resolve_references(e.papers,mapping))
                state.store.remember_aliases(duplicate['id'],aliases)
                state.engine=new_engine
                return {'id':duplicate['id'],'added':False,'warning':warning}
            mapping.update(dict.fromkeys(aliases,p['id']))
            new_engine=MiningEngine(resolve_references(e.papers+[p],mapping))
            state.store.persist_import(p,aliases)
            state.engine=new_engine
        return {'id':p['id'],'added':True,'warning':warning,'workspace_papers':len(new_engine.ids)}
    return router
