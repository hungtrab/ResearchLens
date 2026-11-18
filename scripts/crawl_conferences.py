"""Resumable public metadata crawl: official bulk indexes + OpenAlex source filters.
No PDF bulk downloads. Missing/pending editions are recorded, never invented.
"""
import argparse,concurrent.futures,gzip,hashlib,html,json,os,re,time,uuid
from datetime import datetime,timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin,urlsplit,parse_qs
import xml.etree.ElementTree as ET
import httpx
from researchlens.catalog import CONFERENCES,title_key
from researchlens.sources import SOURCES,WORK_FIELDS,normalize_work
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/conferences';RAW=OUT/'raw'

class Node:
    def __init__(self,tag='',attrs=None):self.tag=tag;self.attrs=dict(attrs or []);self.children=[]
    def text(self):return ''.join(c.text() if isinstance(c,Node) else c for c in self.children)
    def find(self,tag=None,cls=None):
        for child in self.children:
            if isinstance(child,Node):
                if (tag is None or child.tag==tag) and (cls is None or cls in child.attrs.get('class','').split()):yield child
                yield from child.find(tag,cls)
class DOM(HTMLParser):
    def __init__(self,text):
        super().__init__(convert_charrefs=True);self.root=Node();self.stack=[self.root];self.feed(text)
    def handle_starttag(self,tag,attrs):
        node=Node(tag,attrs);self.stack[-1].children.append(node)
        if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:self.stack.append(node)
    def handle_endtag(self,tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].tag==tag:self.stack=self.stack[:i];break
    def handle_data(self,data):self.stack[-1].children.append(data)

def fetch(url):
    RAW.mkdir(parents=True,exist_ok=True);cache=RAW/(hashlib.sha256(url.encode()).hexdigest()+'.gz')
    if cache.exists():return gzip.decompress(cache.read_bytes()).decode('utf-8')
    headers={'User-Agent':'ResearchLens/1.2 (university research metadata indexing)'}
    if url.startswith('https://api.openalex.org/') and os.getenv('OPENALEX_API_KEY'):headers['Authorization']='Bearer '+os.environ['OPENALEX_API_KEY']
    for attempt in range(3):
        try:
            r=httpx.get(url,headers=headers,timeout=90,follow_redirects=True)
            if r.status_code in (429,500,502,503,504) and attempt<2:time.sleep(2**attempt);continue
            r.raise_for_status();content=r.content
            if r.status_code!=200 or b'anubis_challenge' in content[:15000]:raise ValueError('Source requires interactive access')
            temp=cache.with_suffix('.'+uuid.uuid4().hex+'.tmp');temp.write_bytes(gzip.compress(content));temp.replace(cache);return content.decode('utf-8')
        except httpx.TransportError:
            if attempt==2:raise
            time.sleep(2**attempt)
    raise RuntimeError('Retry budget exhausted')

def base(conference,year,id,title,url,**extra):
    return dict(id=id,title=html.unescape(title).strip(),abstract='',year=year,authors=[],venue=f'{conference} {year}',doi=None,url=url,pdf_url=None,
                open_access=True,topics=[conference],collection_topic=conference,citations=0,references=[],source='Official proceedings',conference=conference,**extra)

def virtual(conference,year):
    domain='iclr.cc' if conference=='ICLR' else 'neurips.cc';slug=conference.lower()
    page=f'https://{domain}/virtual/{year}/papers.html';markup=fetch(page)
    links=re.findall(r'["\']([^"\']+orals-posters[^"\']*\.json)["\']',markup)
    url=urljoin(page,links[0]) if links else f'https://{domain}/static/virtual/data/{slug}-{year}-orals-posters.json'
    data=json.loads(fetch(url));rows=data.get('results',data) if isinstance(data,dict) else data
    if not isinstance(rows,list):raise ValueError('Unexpected conference index schema')
    result=[]
    for x in rows:
        if not x.get('name') or x.get('eventtype','').lower() not in ('poster','oral','spotlight'):continue
        if 'workshop' in (x.get('sourceurl') or '').lower():continue
        if x.get('children_ids') or '/session/' in (x.get('virtualsite_url') or ''):continue
        media=next((m.get('uri') for m in x.get('eventmedia',[]) if (m.get('name') or '').lower() in ('paper','openreview') and m.get('uri')),None)
        paper_url=x.get('paper_url') or media or x.get('paper_pdf_url') or (urljoin(page,x['virtualsite_url']) if x.get('virtualsite_url') else page)
        identity=paper_url if paper_url!=page else str(x.get('id') or x.get('uid') or title_key(x['name']))
        key=hashlib.sha1(identity.encode()).hexdigest()[:20]
        if paper_url==page and x.get('id'):paper_url=f'https://{domain}/virtual/{year}/{x["eventtype"].lower()}/{x["id"]}'
        p=base(conference,year,f'{conference}-{year}-{key}',x['name'],paper_url)
        p.update(abstract=re.sub('<[^>]+>',' ',x.get('abstract') or ''),authors=[a['fullname'] for a in x.get('authors',[]) if a.get('fullname')],pdf_url=x.get('paper_pdf_url'),source=f'{conference} official program',track=x.get('sourceurl'),metadata_url=url)
        if p['pdf_url'] and p['pdf_url'].endswith('-Abstract.html'):
            p['pdf_url']=p['pdf_url'].replace('/hash/','/file/').replace('-Abstract.html','-Paper.pdf')
        candidate=p['pdf_url'] or paper_url
        if urlsplit(candidate).hostname=='openreview.net' and parse_qs(urlsplit(candidate).query).get('id'):
            p['pdf_url']='https://openreview.net/pdf?id='+parse_qs(urlsplit(candidate).query)['id'][0]
        result.append(p)
    if not result:raise ValueError('No paper events published in the official index')
    return result,[url]

def pmlr(year):
    # Resolve the volume from the official series index rather than assuming newest IDs.
    home=DOM(fetch('https://proceedings.mlr.press/')).root
    matches=[a for a in home.find('a') if (f'ICML {year}' in a.text() or f'Machine Learning {year}' in a.text()) and re.search(r'v\d+',a.attrs.get('href',''))]
    known={2020:119,2021:139,2022:162,2023:202,2024:235,2025:267,2026:306}
    url=urljoin('https://proceedings.mlr.press/',matches[0].attrs['href']) if matches else f'https://proceedings.mlr.press/v{known[year]}/'
    markup=fetch(url);dom=DOM(markup).root
    heading=' '.join(x.text() for x in dom.find('title'))
    if 'Machine Learning' not in heading and 'ICML' not in heading:raise ValueError('Volume is not confirmed as ICML')
    result=[]
    for row in dom.find('div','paper'):
        title=next(row.find('p','title'),None);authors=next(row.find('span','authors'),None);links=list(row.find('a'))
        abstract_link=next((a.attrs.get('href') for a in links if a.text().strip()=='abs'),None)
        if title is None or not abstract_link:continue
        key=urlsplit(abstract_link).path.strip('/').replace('/','-').removesuffix('.html')
        p=base('ICML',year,'PMLR-'+key,title.text(),abstract_link)
        p.update(authors=[x.strip() for x in (authors.text() if authors else '').replace('\xa0',' ').split(',') if x.strip()],pdf_url=next((a.attrs.get('href') for a in links if a.text().strip()=='Download PDF'),None),source='PMLR',metadata_url=url)
        result.append(p)
    if not result:raise ValueError('No ICML papers found in volume')
    return result,[url]

def anthology(conference,year,tree):
    prefix=f'data/xml/{year}.';slug=conference.lower()
    paths=[x['path'] for x in tree['tree'] if x['path'] in (f'{prefix}{slug}.xml',f'{prefix}findings.xml')]
    result=[];urls=[]
    for path in paths:
        url=f'https://raw.githubusercontent.com/acl-org/acl-anthology/{tree["sha"]}/{path}';urls.append(url)
        root=ET.fromstring(fetch(url));collection=root.attrib['id']
        for volume in root.findall('volume'):
            if collection.endswith('.findings') and volume.attrib.get('id')!=slug:continue
            for paper in volume.findall('paper'):
                if paper.attrib.get('id')=='0':continue
                pid=f'{collection}-{volume.attrib["id"]}.{paper.attrib["id"]}';title=paper.find('title')
                if title is None:continue
                p=base(conference,year,'ACL-'+pid,''.join(title.itertext()),f'https://aclanthology.org/{pid}/')
                abstract=paper.find('abstract');doi=paper.findtext('doi')
                p.update(authors=[' '.join(' '.join(a.itertext()).split()) for a in paper.findall('author')],abstract=''.join(abstract.itertext()) if abstract is not None else '',doi='https://doi.org/'+doi if doi else None,pdf_url=f'https://aclanthology.org/{pid}.pdf',source='ACL Anthology',metadata_url=url,track=volume.attrib['id'])
                result.append(p)
    return result,urls

def cvf(year):
    url=f'https://openaccess.thecvf.com/CVPR{year}?day=all';markup=fetch(url)
    if year==2020:
        url=f'https://openaccess.thecvf.com/CVPR{year}';markup=fetch(url)
    urls=[url]
    if not re.search(r'<dt\s+class=["\']ptitle["\']>',markup):
        days=list(dict.fromkeys(urljoin(url,a.attrs['href']) for a in DOM(markup).root.find('a') if re.fullmatch(r'CVPR2020\.py\?day=2020-06-\d{2}',a.attrs.get('href',''))))
        if days:markup='\n'.join(fetch(day) for day in days);urls+=days
    pieces=re.split(r'<dt\s+class=["\']ptitle["\']>',markup)[1:];result=[]
    for piece in pieces:
        dom=DOM(piece).root;link=next((a for a in dom.find('a') if '/html/' in a.attrs.get('href','')),None)
        if not link:continue
        paper_url=urljoin(url,link.attrs['href']);p=base('CVPR',year,'CVF-'+hashlib.sha1(paper_url.encode()).hexdigest()[:20],link.text(),paper_url)
        p.update(authors=[a.attrs.get('value','') for a in dom.find('input') if a.attrs.get('name')=='query_author'],pdf_url=next((urljoin(url,a.attrs['href']) for a in dom.find('a') if a.attrs.get('href','').endswith('_paper.pdf')),None),source='CVF Open Access',metadata_url=url)
        result.append(p)
    if not result:raise ValueError('No CVPR paper entries published in the index')
    return list({p['id']:p for p in result}.values()),urls

def openalex(conference,start,end):
    rows=[];urls=[];cursor='*';seen=set()
    while cursor:
        params={'filter':f'primary_location.source.id:{"|".join(SOURCES[conference])},publication_year:{start}-{end}','select':WORK_FIELDS,'per-page':200,'cursor':cursor}
        url=str(httpx.URL('https://api.openalex.org/works',params=params));data=json.loads(fetch(url));urls.append(url)
        for w in data['results']:
            if not w.get('title'):continue
            p=normalize_work(w);p.update(conference=conference,collection_topic=conference,metadata_url='https://api.openalex.org/works/'+p['id']);rows.append(p)
        print(f'OpenAlex {conference}: {len(rows)}/{data["meta"]["count"]}',flush=True)
        if not data['results']:break
        cursor=data['meta'].get('next_cursor')
        if cursor in seen:raise ValueError('Source repeated its cursor')
        seen.add(cursor);time.sleep(.2)
    return rows,urls

def merge_records(groups):
    # OpenAlex first: retain its IDs so references stay resolvable in the mining engine.
    records={};identities={}
    ordered=sorted((p for group in groups for p in group),key=lambda p:(p['source']!='OpenAlex',p['id']))
    for p in ordered:
        identity=(p['conference'],p['year'],title_key(p['title']))
        key=identities.get(identity)
        if key:
            old=records[key];old.setdefault('alternate_ids',[])
            if p['id']!=key and p['id'] not in old['alternate_ids']:old['alternate_ids'].append(p['id'])
            old.setdefault('provenance_urls',[old.get('metadata_url',old['url'])])
            if p.get('metadata_url') and p['metadata_url'] not in old['provenance_urls']:old['provenance_urls'].append(p['metadata_url'])
            for field in ['abstract','authors','doi','pdf_url']:
                if not old.get(field) and p.get(field):old[field]=p[field]
            if p['source']!='OpenAlex':old.update(venue=f'{p["conference"]} {p["year"]}',official_url=p['url'],track=p.get('track'))
        else:
            if p['id'] in records:
                # The same event can be repeated in virtual program sessions.
                continue
            records[p['id']]=p;identities[identity]=p['id']
    return sorted(records.values(),key=lambda p:(p['conference'],p['year'],p['id']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--from-year',type=int,default=2020);ap.add_argument('--to-year',type=int,default=2026);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    if not 2020<=args.from_year<=args.to_year<=2026:ap.error('Current official adapters support 2020–2026')
    OUT.mkdir(parents=True,exist_ok=True)
    tree=json.loads(fetch('https://api.github.com/repos/acl-org/acl-anthology/git/trees/master?recursive=1'))
    if tree.get('truncated'):raise RuntimeError('ACL repository index is truncated')
    jobs=[]
    for c in CONFERENCES:
        jobs.append((c,None,'OpenAlex',lambda c=c:openalex(c,args.from_year,args.to_year)))
        if c=='AAAI':continue
        for y in range(args.from_year,args.to_year+1):
            fn=(lambda c=c,y=y:virtual(c,y)) if c in ('ICLR','NeurIPS') else (lambda y=y:pmlr(y)) if c=='ICML' else (lambda y=y:cvf(y)) if c=='CVPR' else (lambda c=c,y=y:anthology(c,y,tree))
            jobs.append((c,y,'Official',fn))
    groups=[];statuses=[]
    def run(job):
        c,y,source,fn=job
        try:
            rows,urls=fn();return rows,dict(conference=c,year=y,source=source,status='fetched' if rows else 'no_published_records',records=len(rows),urls=urls)
        except Exception as exc:return [],dict(conference=c,year=y,source=source,status='unavailable',records=0,error=str(exc)[:400])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rows,status in pool.map(run,jobs):
            groups.append(rows);statuses.append(status);print(json.dumps({k:v for k,v in status.items() if k!='urls'}),flush=True)
            (OUT/'progress.json').write_text(json.dumps(statuses,indent=2))
    records=merge_records(groups);target=OUT/'papers.jsonl';temp=target.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in records));temp.replace(target)
    coverage=[dict(conference=c,year=y,papers=sum(p['conference']==c and p['year']==y for p in records)) for c in CONFERENCES for y in range(args.from_year,args.to_year+1)]
    manifest=dict(collected_at=datetime.now(timezone.utc).isoformat(),years=[args.from_year,args.to_year],papers=len(records),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),acl_commit=tree['sha'],coverage=coverage,sources=statuses,
        scope='All records returned by completed official bulk indexes and exhausted OpenAlex conference-source cursors in the selected years. Coverage is not a claim that all accepted papers are indexed; missing/unpublished editions and source failures are explicit. No bulk PDF download. Exact normalized title dedup within conference/year; OpenAlex IDs retained when matched.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('TOTAL',len(records),flush=True)
if __name__=='__main__':main()
