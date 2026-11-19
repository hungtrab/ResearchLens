from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
import json
import os
import logging
from zipfile import BadZipFile
import secrets
import time
import threading
from collections import defaultdict
from typing import Literal
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from researchlens.mining import MiningEngine, extract, opinion
from researchlens.decision import DecisionModel
from researchlens.reviews import deduplicate, train_model, predict, rating_label
from researchlens.store import Store
from researchlens.passages import PassageIndex
from researchlens.workspace import workspace_router
from researchlens.catalog import Catalog
from researchlens.sources import OpenAlex
from researchlens.discovery import discovery_router,resolve_references

ROOT=Path(__file__).resolve().parents[1]
class TextInput(BaseModel):
    text: str=Field(min_length=1,max_length=30000)
class RecommendationInput(BaseModel):
    seeds: list[str]=Field(default_factory=list,max_length=10000)
    method: Literal['fusion','content','graph','hybrid','popular','cocitation','decision']='fusion'
    diversity: float=Field(default=.15,ge=0,le=1)
    limit: int=Field(default=12,ge=1,le=50)


def create_app(data_dir=None,db_path=None):
    data=Path(data_dir or os.environ.get('RESEARCHLENS_DATA',ROOT/'data'))
    @asynccontextmanager
    async def lifespan(app):
        path=data/'papers.jsonl'
        if not path.exists(): raise RuntimeError('Missing corpus. Run: uv run python scripts/collect.py')
        papers=[json.loads(line) for line in path.read_text().split('\n') if line.strip()]
        app.state.store=Store(db_path or os.environ.get('RESEARCHLENS_DB',ROOT/'data/researchlens.db'))
        app.state.store.ingest(papers)
        app.state.benchmark_count=len(papers)
        app.state.passages = None
        passage_path = data / 'passages.npz'
        if passage_path.exists():
            try:
                app.state.passages = PassageIndex.load(passage_path, path)
            except (ValueError, KeyError, OSError, BadZipFile):
                logging.getLogger(__name__).warning("Passage index unavailable. Rebuild it for this corpus.", exc_info=True)
        base_ids={p['id'] for p in papers}
        imported=[p for p in app.state.store.imported() if p['id'] not in base_ids]
        app.state.engine=MiningEngine(resolve_references(papers+imported,app.state.store.aliases()))
        app.state.decision=None
        if (ROOT/'artifacts/decision_model.pkl').exists() and (ROOT/'artifacts/specter.npz').exists():
            app.state.decision=DecisionModel.load(ROOT/'artifacts/decision_model.pkl',ROOT/'artifacts/specter.npz')
        app.state.import_lock=threading.Lock()
        app.state.online=OpenAlex()
        app.state.data_dir=data
        app.state.catalog=Catalog(Path(app.state.store.path).with_name('conference-catalog.sqlite'))
        app.state.catalog.sync(data/'conferences/papers.jsonl')
        path=data/'reviews.jsonl'
        reviews=deduplicate([json.loads(line) for line in path.read_text().split('\n') if line.strip()]) if path.exists() else []
        app.state.reviews=defaultdict(list)
        for r in reviews: app.state.reviews[r['paper_id']].append(r)
        app.state.model=train_model(reviews)
        app.state.review_count=len(reviews)
        yield
    app=FastAPI(title='ResearchLens',version='1.2.0',description='Academic paper discovery and web mining. Local teaching deployment.',lifespan=lifespan,docs_url=None,redoc_url=None)

    @app.middleware('http')
    async def session(request:Request,call_next):
        profile=request.cookies.get('rl_profile','')
        valid=len(profile)==64 and all(c in '0123456789abcdef' for c in profile)
        request.state.profile=profile if valid else secrets.token_hex(32)
        # Local UI writes require same-origin; arbitrary scripts may use API via non-browser clients.
        if request.method in {'POST','DELETE','PUT','PATCH'}:
            origin=request.headers.get('origin')
            if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):
                return PlainTextResponse('Cross-origin writes are disabled.',status_code=403)
        response=await call_next(request)
        if not valid: response.set_cookie('rl_profile',request.state.profile,httponly=True,samesite='strict',max_age=60*60*24*365,secure=request.url.scheme=='https')
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='strict-origin-when-cross-origin'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-src 'self' blob:; object-src 'self' blob:; base-uri 'self'; frame-ancestors 'none'"
        if request.url.path.startswith('/api/'): response.headers['Cache-Control']='no-store'
        return response

    @app.get('/api/health')
    def health(): return {'status':'ok','papers':len(app.state.engine.ids),'review_model':app.state.model is not None,'solr_enabled':bool(os.environ.get('SOLR_URL'))}
    @app.get('/api/search')
    def search(request:Request,q:str=Query('',max_length=500),method:Literal['bm25','tfidf','hybrid','pagerank','solr']='hybrid',
               year_min:int|None=Query(None,ge=1800,le=2100),year_max:int|None=Query(None,ge=1800,le=2100),
               topic:str|None=None,venue:str|None=None,open_access:bool=False,offset:int=Query(0,ge=0),limit:int=Query(20,ge=1,le=100)):
        if year_min is not None and year_max is not None and year_min>year_max: raise HTTPException(422,'Start year must not exceed end year.')
        start=time.perf_counter()
        try:
            if method=='solr':
                from researchlens.solr import SolrSearch, expression
                url=os.environ.get('SOLR_URL')
                if not url: raise HTTPException(503,'Solr is not enabled. Set SOLR_URL and index the corpus.')
                expression(q)  # Validate user syntax before entering the upstream service boundary.
                try: result=SolrSearch(url).search(app.state.engine,q,year_min,year_max,topic,venue,open_access,offset,limit)
                except Exception as exc:
                    raise HTTPException(503,'Solr is unavailable or not indexed. Check the Solr service and index.') from exc
            else: result=app.state.engine.search(q,method,year_min,year_max,topic,venue,open_access,offset,limit)
        except ValueError as e: raise HTTPException(422,str(e))
        result['elapsed_ms']=round((time.perf_counter()-start)*1000,2)
        result['method']=method; result['offset']=offset
        if q.strip(): app.state.store.remember(request.state.profile,q)
        return result
    @app.get('/api/papers/{pid}')
    def paper(pid:str):
        e=app.state.engine
        if pid not in e.pos: raise HTTPException(404,'Paper not found')
        p=e.by_id[pid]
        return dict(p,metrics=e.summary(e.pos[pid]),keywords=e.keywords(pid),entities=extract(p['title']+' '+p.get('abstract','')),
                    references=[e.summary(e.pos[r]) for r in e.graph.successors(pid)],cited_by=[e.summary(e.pos[r]) for r in e.graph.predecessors(pid)],
                    review_count=len(app.state.reviews[pid]))
    @app.get('/api/papers/{pid}/reviews')
    def paper_reviews(pid:str):
        if pid not in app.state.engine.pos: raise HTTPException(404,'Paper not found')
        return {'items':[dict(r,observed_label=rating_label(r['rating']),analysis=opinion(r['text']),prediction=predict(app.state.model,r['text'])) for r in app.state.reviews[pid]],
                'note':'Reviewer opinions are subjective. Predicted polarity is not a quality or acceptance judgment.'}
    @app.get('/api/reviews')
    def reviews(limit:int=Query(20,ge=1,le=100),offset:int=Query(0,ge=0)):
        rows=[r for rs in app.state.reviews.values() for r in rs]
        return {'total':len(rows),'items':[dict(r,title=app.state.engine.by_id[r['paper_id']]['title'],observed_label=rating_label(r['rating'])) for r in rows[offset:offset+limit]]}
    @app.post('/api/opinion')
    def analyze(body:TextInput): return dict(opinion(body.text),prediction=predict(app.state.model,body.text))
    @app.post('/api/extract')
    def extraction(body:TextInput): return extract(body.text)
    @app.get('/api/graph')
    def graph(paper_id:str|None=None,limit:int=Query(70,ge=1,le=200)):
        try: return app.state.engine.subgraph(paper_id,limit)
        except KeyError: raise HTTPException(404,'Paper not found')
    @app.get('/api/stats')
    def stats(): return dict(app.state.engine.stats(),reviews=app.state.review_count,benchmark_papers=app.state.benchmark_count)
    @app.get('/api/facets')
    def facets():
        e=app.state.engine
        return {'topics':sorted({p['collection_topic'] for p in e.papers}),'venues':sorted({p['venue'] for p in e.papers}),
                'years':sorted({p['year'] for p in e.papers if p.get('year')},reverse=True)}
    @app.post('/api/recommendations')
    def recommendations(body:RecommendationInput):
        try: return app.state.engine.recommend(body.seeds,body.limit,body.method,body.diversity,app.state.decision)
        except ValueError as e: raise HTTPException(422,str(e))
    @app.get('/api/library')
    def library(request:Request):
        e=app.state.engine; ids=[p for p in app.state.store.saved(request.state.profile) if p in e.pos]
        return {'ids':ids,'items':[e.summary(e.pos[p]) for p in ids if p in e.pos],'history':app.state.store.history(request.state.profile)}
    @app.post('/api/library/{pid}')
    def save(pid:str,request:Request):
        if pid not in app.state.engine.pos: raise HTTPException(404,'Paper not found')
        app.state.store.save(request.state.profile,pid); return {'saved':True}
    @app.delete('/api/library/{pid}')
    def remove(pid:str,request:Request):
        app.state.store.remove(request.state.profile,pid); return {'saved':False}
    @app.get('/api/library/export')
    def export(request:Request,format:Literal['bibtex','json']='bibtex'):
        e=app.state.engine; papers=[e.by_id[p] for p in app.state.store.saved(request.state.profile) if p in e.by_id]
        if format=='json': return Response(json.dumps(papers,ensure_ascii=False,indent=2),media_type='application/json',headers={'Content-Disposition':'attachment; filename="researchlens-library.json"'})
        def escape(s):
            table={'\\':r'\textbackslash{}','{':r'\{','}':r'\}','%':r'\%','&':r'\&','_':r'\_','#':r'\#','$':r'\$','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
            return ''.join(table.get(c,c) for c in str(s)).replace('\n',' ')
        output=[]
        for p in papers:
            fields={'title':p['title'],'author':' and '.join(p.get('authors',[])),'year':p.get('year',''),'journal':p.get('venue',''),'url':p.get('url','')}
            if p.get('doi'): fields['doi']=p['doi'].replace('https://doi.org/','')
            output.append('@article{'+p['id']+',\n'+',\n'.join('  '+k+' = {'+escape(v)+'}' for k,v in fields.items())+'\n}')
        return Response('\n\n'.join(output),media_type='application/x-bibtex',headers={'Content-Disposition':'attachment; filename="researchlens-library.bib"'})
    @app.get('/api/experiments')
    def experiments():
        path=ROOT/'artifacts/results.json'
        return json.loads(path.read_text()) if path.exists() else {'status':'not_run','command':'uv run python scripts/evaluate.py'}
    app.include_router(workspace_router())
    app.include_router(discovery_router())
    app.mount('/static',StaticFiles(directory=Path(__file__).parent/'static'),name='static')
    @app.get('/docs',include_in_schema=False)
    def docs(): return FileResponse(Path(__file__).parent/'static/api-docs.html')
    @app.get('/')
    def index(): return FileResponse(Path(__file__).parent/'static/index.html')
    return app
app=create_app()
