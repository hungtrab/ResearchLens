"""Fetch public metadata; cache responses and pin provenance. No HTML scraping."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, re, time
from datetime import datetime, timezone
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
TOPICS = ['graph neural networks', 'information retrieval', 'recommender systems', 'sentiment analysis', 'large language models', 'deep learning', 'reinforcement learning', 'knowledge graphs']

def get(url, params=None):
    for attempt in range(5):
        try:
            r = httpx.get(url, params=params, timeout=60, follow_redirects=True,
                          headers={'User-Agent':'ResearchLens/1.0 (academic metadata research)'})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 15)); continue
            r.raise_for_status()
            return r.json()
        except httpx.TransportError:
            if attempt == 4: raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f'Exhausted retries: {url}')

def abstract(index):
    if not index: return ''
    words = {i: w for w, positions in index.items() for i in positions}
    return ' '.join(words[i] for i in sorted(words))

def title_key(s):
    return re.sub(r'\W+', '', s.casefold())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--per-topic',type=int,default=400); ap.add_argument('--skip-reviews',action='store_true'); args=ap.parse_args()
    if not 1<=args.per_topic<=10000: ap.error('--per-topic must be between 1 and 10000')
    RAW.mkdir(parents=True, exist_ok=True)
    papers={}
    for topic in TOPICS:
        for page in range(1, (args.per_topic+199)//200+1):
            cache=RAW/f'openalex-{topic.replace(" ","-")}-{page}.json'
            if not cache.exists():
                params={'search':topic,'filter':'language:en,to_publication_date:2025-12-31','per-page':200,'page':page,
                        'select':'id,doi,title,publication_year,authorships,primary_location,abstract_inverted_index,referenced_works,cited_by_count,topics,open_access'}
                if os.getenv('OPENALEX_API_KEY'): params['api_key']=os.environ['OPENALEX_API_KEY']
                cache.write_text(json.dumps(get('https://api.openalex.org/works',params)),encoding='utf8')
                time.sleep(.3)
            for w in json.loads(cache.read_text())['results'][:min(200,args.per_topic-(page-1)*200)]:
                if not w.get('title'): continue
                pid=w['id'].rsplit('/',1)[-1]
                if pid in papers: continue
                loc=w.get('primary_location') or {}; source=loc.get('source') or {}
                papers[pid]={'id':pid,'title':w['title'],'abstract':abstract(w.get('abstract_inverted_index')),'year':w.get('publication_year'),
                    'authors':[a['author']['display_name'] for a in w.get('authorships',[])], 'venue':source.get('display_name') or 'Unknown venue',
                    'doi':w.get('doi'),'url':w.get('doi') or w['id'],'pdf_url':loc.get('pdf_url'),'open_access':(w.get('open_access') or {}).get('is_oa',False),
                    'topics':[t['display_name'] for t in w.get('topics',[])], 'collection_topic':topic,'citations':w.get('cited_by_count',0),
                    'references':[x.rsplit('/',1)[-1] for x in w.get('referenced_works',[])], 'source':'OpenAlex'}
            print(topic,page,len(papers),flush=True)
    reviews=[]
    if not args.skip_reviews:
        manifest=RAW/'peerread-tree.json'
        if not manifest.exists(): manifest.write_text(json.dumps(get('https://api.github.com/repos/allenai/PeerRead/git/trees/master',{'recursive':1})))
        tree=json.loads(manifest.read_text()); sha=tree['sha']
        files=[x['path'] for x in tree['tree'] if x['path'].startswith('data/iclr_2017/') and '/reviews/' in x['path'] and x['path'].endswith('.json')]
        def fetch(path):
            cache=RAW/('peerread-'+path.split('/')[2]+'-'+Path(path).name)
            if not cache.exists(): cache.write_text(json.dumps(get(f'https://raw.githubusercontent.com/allenai/PeerRead/{sha}/{path}')),encoding='utf8')
            return path,json.loads(cache.read_text())
        keys={title_key(p['title']):pid for pid,p in papers.items()}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            for path,p in pool.map(fetch,files):
                key=title_key(p.get('title','')); pid=keys.get(key) or 'PR'+Path(path).stem
                if pid not in papers:
                    papers[pid]={'id':pid,'title':p['title'],'abstract':p.get('abstract',''),'year':2017,'authors':[a.strip() for a in p.get('authors','').split(',') if a.strip()] if isinstance(p.get('authors'),str) else p.get('authors',[]),
                        'venue':'ICLR 2017 submissions','doi':None,'url':f'https://github.com/allenai/PeerRead/blob/{sha}/{path}',
                        'pdf_url':None,'open_access':False,'topics':['Machine Learning'],'collection_topic':'peer review', 'citations':0,'references':[], 'source':'PeerRead'}
                for i,r in enumerate(p.get('reviews',[])):
                    score=r.get('RECOMMENDATION'); body=r.get('comments','').strip()
                    # Only scored reviews: exclude author replies, meta reviews and public comments.
                    if score is None or not body or r.get('IS_META_REVIEW'): continue
                    try: score=float(score)
                    except (TypeError,ValueError): continue
                    reviews.append({'id':f'{pid}-r{i}','paper_id':pid,'text':body,'rating':score,'split':path.split('/')[2],
                        'source':'PeerRead ICLR 2017','source_url':f'https://github.com/allenai/PeerRead/blob/{sha}/{path}'})
        from researchlens.reviews import deduplicate
        reviews=deduplicate(reviews)
        print('Unique scored reviews:',len(reviews),flush=True)
    for name,rows in [('papers',list(papers.values())),('reviews',reviews)]:
        target=ROOT/f'data/{name}.jsonl'
        temp=target.with_suffix('.jsonl.tmp')
        temp.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in rows),encoding='utf8')
        temp.replace(target)
    provenance={'collected_at':datetime.now(timezone.utc).isoformat(),'papers':len(papers),'reviews':len(reviews),'topics':TOPICS,
        'selection':f'First {args.per_topic} relevance-ranked OpenAlex works per query, English, publication date <= 2025-12-31; union with PeerRead ICLR 2017.',
        'openalex':'https://api.openalex.org/works','peerread_commit':None if args.skip_reviews else sha,
        'files':{name:hashlib.sha256((ROOT/f'data/{name}.jsonl').read_bytes()).hexdigest() for name in ('papers','reviews')}}
    (ROOT/'data/provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2))
if __name__=='__main__': main()
