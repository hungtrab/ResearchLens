"""Create explicit schema and replace the ResearchLens Solr collection contents."""
import json,os,time
from pathlib import Path
import httpx
from researchlens.mining import MiningEngine
ROOT=Path(__file__).resolve().parents[1]
def main():
    url=os.environ.get('SOLR_URL','http://127.0.0.1:8983/solr/researchlens').rstrip('/')
    papers=[json.loads(x) for x in open(ROOT/'data/papers.jsonl',encoding='utf8') if x.strip()];e=MiningEngine(papers)
    with httpx.Client(timeout=60) as c:
        existing=c.get(url+'/schema/fields');existing.raise_for_status();names={f['name'] for f in existing.json()['fields']}
        fields={'p_title':'text_en','p_abstract':'text_en','p_year':'pint','p_topic':'string','p_venue':'string','p_oa':'boolean','p_pagerank':'pdouble'}
        for name,kind in fields.items():
            if name not in names:
                r=c.post(url+'/schema',json={'add-field':{'name':name,'type':kind,'stored':True,'indexed':True,'multiValued':False}});r.raise_for_status()
        docs=[{'id':p['id'],'p_title':p['title'],'p_abstract':p.get('abstract',''),'p_year':p.get('year'),'p_topic':p['collection_topic'],'p_venue':p['venue'],'p_oa':p['open_access'],'p_pagerank':e.pagerank[p['id']]} for p in papers]
        # Purpose-built collection only; do not point this script at an unrelated Solr core.
        r=c.post(url+'/update',json={'delete':{'query':'*:*'}});r.raise_for_status()
        for start in range(0,len(docs),400):
            r=c.post(url+'/update',json=docs[start:start+400]);r.raise_for_status()
        r=c.post(url+'/update',params={'commit':'true'},json={});r.raise_for_status()
        result=c.get(url+'/select',params={'q':'*:*','rows':0});result.raise_for_status()
        print(json.dumps({'indexed':result.json()['response']['numFound'],'collection':url}))
if __name__=='__main__':main()
