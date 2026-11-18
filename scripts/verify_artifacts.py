"""Independently check saved data/splits/aggregate metrics against their artifacts."""
import hashlib,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from researchlens.reviews import deduplicate
from researchlens.catalog import title_key
ROOT=Path(__file__).resolve().parents[1]
def rows(path):
 with open(ROOT/path,encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def main():
 r=json.load(open(ROOT/'artifacts/results.json'));p=rows('data/papers.jsonl');v=rows('data/reviews.jsonl');ids={x['id'] for x in p};by={x['id']:x for x in p}
 assert len(ids)==len(p),'Duplicate paper IDs'
 assert all(x['paper_id'] in ids for x in v),'Orphan review'
 assert len(deduplicate(v))==len(v),'Duplicate reviews'
 assert r['corpus_sha256']==sha('data/papers.jsonl'),'Corpus changed since evaluation'
 assert r['reviews_sha256']==sha('data/reviews.jsonl'),'Reviews changed since evaluation'
 provenance=json.load(open(ROOT/'data/provenance.json'))
 for name in ('papers','reviews'):assert provenance['files'][name]==sha(f'data/{name}.jsonl')
 splits={s:{x['paper_id'] for x in v if x['split']==s} for s in ['train','dev','test']}
 assert not(splits['train']&splits['test'] or splits['train']&splits['dev'] or splits['dev']&splits['test'])
 held=set();tasks=rows('artifacts/recommendation_split.jsonl')
 for t in tasks:
  assert not(set(t['seeds'])&set(t['relevant']))
  assert set(t['seeds'])|set(t['relevant'])==set(by[t['paper_id']]['references'])&ids-{t['paper_id']}
  held.update((t['paper_id'],d) for d in t['relevant'])
 assert len(held)==r['recommendation']['masked_edges']
 assert len(tasks)==r['recommendation']['tasks']
 assert r['recommendation']['split_sha256']==sha('artifacts/recommendation_split.jsonl')
 for task,path,metrics in [('retrieval','retrieval_per_query.jsonl',['ndcg_at_10','map','precision_at_10','recall_at_10','mrr']),('recommendation','recommendation_per_paper.jsonl',['ndcg_at_10','recall_at_10','mrr','diversity'])]:
  records=rows('artifacts/'+path)
  for summary in r[task]['summary']:
   group=[x for x in records if x['method']==summary['method']]
   assert len(group)==summary['n']
   for metric in metrics:assert np.isclose(np.mean([x[metric] for x in group]),summary[metric],atol=1e-12)
 conference_check={}
 manifest_path=ROOT/'data/conferences/manifest.json'
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text());conference=rows('data/conferences/papers.jsonl')
  assert manifest['sha256']==sha('data/conferences/papers.jsonl')
  assert len(conference)==manifest['papers']==len({x['id'] for x in conference})
  assert len({(x['conference'],x['year'],title_key(x['title'])) for x in conference})==len(conference)
  counts=Counter((x['conference'],x['year']) for x in conference)
  assert all(counts[x['conference'],x['year']]==x['papers'] for x in manifest['coverage'])
  measured=json.loads((ROOT/'artifacts/catalog_results.json').read_text())
  assert measured['source_sha256']==manifest['sha256'] and measured['papers']==len(conference)
  assert measured['abstracts']==sum(bool(x['abstract']) for x in conference)
  assert measured['latency']['queries']==len(measured['measurements'])==100
  for percentile in (50,95):assert np.isclose(measured['latency'][f'p{percentile}_ms'],np.percentile([x['elapsed_ms'] for x in measured['measurements']],percentile))
  conference_check={'conference_papers':len(conference),'coverage_cells':len(manifest['coverage']),'catalog_latency_aggregates':True}
 print(json.dumps({'verified':True,'papers':len(p),'reviews':len(v),'reference_tasks':len(tasks),'masked_edges':len(held),'checks':['IDs','orphan reviews','deduplication','data hashes','split disjointness','target partition','split hash','aggregate metrics'],**conference_check},indent=2))
if __name__=='__main__':main()
