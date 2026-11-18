"""Optional Apache Solr/Lucene retrieval adapter. Host is operator-configured."""
import re
import httpx
from researchlens.mining import query_parts

SPECIAL=re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')
def escape(term):return SPECIAL.sub(r'\\\1',term)
def expression(query):
    clean,excluded,phrases,and_mode=query_parts(query)
    # Quote each input token/phrase; user input cannot inject Solr local parameters.
    import shlex
    terms=[p for p in shlex.split(query) if not p.startswith('-') and p.upper() not in {'AND','OR'}]
    pieces=['"'+escape(t)+'"' for t in terms]
    if not pieces:return '*:*'+''.join(' AND NOT "'+escape(x)+'"' for x in excluded)
    joined=(' AND ' if and_mode else ' OR ').join(pieces)
    return '('+joined+')'+''.join(' AND "'+escape(x)+'"' for x in phrases)+''.join(' AND NOT "'+escape(x)+'"' for x in excluded)

class SolrSearch:
    def __init__(self,url):self.url=url.rstrip('/')
    def search(self,engine,query='',year_min=None,year_max=None,topic=None,venue=None,open_access=False,offset=0,limit=20):
        params={'q':expression(query),'defType':'edismax','qf':'p_title^2 p_abstract','fl':'id,score','rows':limit,'start':offset,'wt':'json','sort':'score desc,id asc' if query.strip() else 'p_pagerank desc,id asc'}
        fq=[]
        if year_min is not None or year_max is not None:fq.append(f'p_year:[{year_min if year_min is not None else "*"} TO {year_max if year_max is not None else "*"}]')
        if topic:fq.append('p_topic:"'+escape(topic)+'"')
        if venue:fq.append('p_venue:"'+escape(venue)+'"')
        if open_access:fq.append('p_oa:true')
        if fq:params['fq']=fq
        r=httpx.get(self.url+'/select',params=params,timeout=15);r.raise_for_status();payload=r.json()['response']
        docs=payload['docs']
        if any(d['id'] not in engine.pos for d in docs):raise RuntimeError('Solr corpus is stale. Re-run scripts/solr_index.py.')
        return {'total':payload['numFound'],'items':[engine.summary(engine.pos[d['id']],float(d.get('score',0))) for d in docs]}
