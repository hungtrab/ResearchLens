"""Public scholarly APIs. Responses are normalized, never executed or treated as instructions."""
import os
import re
import threading
import time
import copy
import httpx
from researchlens.catalog import title_key

OPENALEX='https://api.openalex.org'
WORK_FIELDS='id,doi,title,publication_year,authorships,primary_location,best_oa_location,abstract_inverted_index,referenced_works,cited_by_count,topics,open_access'
SOURCES={
 'ICML':['S4306419644'], 'NeurIPS':['S4363606243'], 'ICLR':['S4306419637'],
 'AAAI':['S4210191458'],'EMNLP':['S4363608991','S4306418267'],
 'NAACL':['S4363608774','S4306420633'],'ACL':['S4363608652'],'CVPR':['S4210176548'],
}
class SourceUnavailable(Exception):pass

def abstract(index):
    if not isinstance(index,dict):return ''
    if any(not isinstance(w,str) or not isinstance(positions,list) for w,positions in index.items()):raise ValueError('Invalid abstract index')
    words={i:w for w,positions in index.items() for i in positions if isinstance(i,int) and 0<=i<100000}
    return ' '.join(words[i] for i in sorted(words))

def normalize_work(w):
    if not isinstance(w,dict):raise ValueError('Invalid OpenAlex paper metadata')
    pid=(w.get('id') or '').rsplit('/',1)[-1]
    if not re.fullmatch(r'W[0-9]+',pid) or not isinstance(w.get('title'),str) or not w['title'].strip():raise ValueError('Invalid OpenAlex paper metadata')
    loc=w.get('primary_location') or {};oa=w.get('best_oa_location') or {};source=loc.get('source') or {}
    for field in ('authorships','referenced_works','topics'):
        if not isinstance(w.get(field,[]),list):raise ValueError('Invalid '+field)
    for a in w.get('authorships',[]):
        if not isinstance(a,dict) or not isinstance(a.get('author'),dict):raise ValueError('Invalid authorship')
    if any(not isinstance(t,dict) for t in w.get('topics',[])):raise ValueError('Invalid topic')
    year=w.get('publication_year')
    if year is not None:
        if isinstance(year,bool) or not re.fullmatch(r'\d{4}',str(year)) or not 1800<=int(year)<=2100:raise ValueError('Invalid publication year')
        year=int(year)
    citations=w.get('cited_by_count') or 0
    if isinstance(citations,bool) or not str(citations).isdigit():raise ValueError('Invalid citation count')
    for value in [w.get('doi'),loc.get('pdf_url'),oa.get('pdf_url'),source.get('display_name')]+[a['author'].get('display_name') for a in w.get('authorships',[])]+[t.get('display_name') for t in w.get('topics',[])]:
        if value is not None and not isinstance(value,str):raise ValueError('Invalid text metadata')
    source_id=(source.get('id') or '').rsplit('/',1)[-1]
    conference=next((c for c,ids in SOURCES.items() if source_id in ids),None)
    return dict(id=pid,title=w['title'].strip(),abstract=abstract(w.get('abstract_inverted_index')),year=year,
        authors=[a.get('author',{}).get('display_name','') for a in w.get('authorships',[]) if a.get('author',{}).get('display_name')],
        venue=source.get('display_name') or 'Unknown venue',doi=w.get('doi'),url=w.get('doi') or w['id'],
        pdf_url=oa.get('pdf_url') or loc.get('pdf_url'),open_access=bool((w.get('open_access') or {}).get('is_oa')),
        topics=[t['display_name'] for t in w.get('topics',[]) if t.get('display_name')],collection_topic=conference or 'online discovery',
        citations=int(citations),references=[x.rsplit('/',1)[-1] for x in w.get('referenced_works',[]) if isinstance(x,str) and re.fullmatch(r'https://openalex.org/W[0-9]+',x)],
        source='OpenAlex',conference=conference,metadata_url=OPENALEX+'/works/'+pid)

class OpenAlex:
    def __init__(self):self.cache={};self.lock=threading.Lock()
    def get(self,path,params=None):
        params=dict(params or {});headers={'User-Agent':'ResearchLens/1.2 (academic metadata discovery)'}
        key=os.getenv('OPENALEX_API_KEY')
        if key:headers['Authorization']='Bearer '+key
        cache_key=(path,tuple(sorted(params.items())))
        with self.lock:
            cached=self.cache.get(cache_key)
            if cached and time.monotonic()-cached[0]<300:return copy.deepcopy(cached[1])
        try:
            response=httpx.get(OPENALEX+path,params=params,headers=headers,timeout=25,follow_redirects=False)
            if response.status_code in (401,403,429):raise SourceUnavailable('OpenAlex access or daily quota is unavailable. Try later or configure a free OPENALEX_API_KEY on the server.')
            if response.status_code==404:raise KeyError('Paper not found at OpenAlex')
            response.raise_for_status();result=response.json()
            if not isinstance(result,dict):raise ValueError('Malformed source response')
        except (httpx.HTTPError,ValueError) as exc:raise SourceUnavailable('OpenAlex is currently unavailable. Local search still works.') from exc
        with self.lock:
            if len(self.cache)>=128:self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key]=(time.monotonic(),result)
        return copy.deepcopy(result)
    def search(self,q,year_min=None,year_max=None,offset=0,limit=20):
        params={'search':q,'per-page':limit,'page':offset//limit+1,'select':WORK_FIELDS}
        filters=[]
        if year_min is not None:filters.append(f'from_publication_date:{year_min}-01-01')
        if year_max is not None:filters.append(f'to_publication_date:{year_max}-12-31')
        if filters:params['filter']=','.join(filters)
        data=self.get('/works',params)
        try:
            return {'items':[normalize_work(w) for w in data['results'] if w.get('title')], 'total':data['meta']['count'],'method':'OpenAlex live search','source_url':'https://openalex.org/works','cached_seconds':300}
        except (AttributeError,KeyError,TypeError,ValueError) as exc:raise SourceUnavailable('OpenAlex returned incomplete search metadata.') from exc
    def paper(self,pid):
        if not re.fullmatch(r'W[0-9]+',pid):raise ValueError('Invalid OpenAlex ID')
        try:return normalize_work(self.get('/works/'+pid,{'select':WORK_FIELDS}))
        except (AttributeError,TypeError,ValueError) as exc:raise SourceUnavailable('OpenAlex returned invalid paper metadata.') from exc
