"""Reproducible evaluation; never invent labels, user ratings, or benchmark results."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, time, zipfile
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import httpx
import numpy as np
import networkx as nx
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.dummy import DummyClassifier
from researchlens.mining import MiningEngine,opinion,unit
from researchlens.reviews import deduplicate,train_model,rating_label,LABELS
from researchlens.evaluation import ranking_metrics,summarize,bootstrap_mean

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts'; DATA=ROOT/'data'
def read_rows(path):
    with open(path,encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def save_rows(name,rows):
    (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def retrieval():
    archive=DATA/'raw/scifact.zip'
    if not archive.exists():
        url='https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
        r=httpx.get(url,timeout=120,follow_redirects=True); r.raise_for_status();archive.write_bytes(r.content)
    if not (DATA/'scifact/corpus.jsonl').exists():
        with zipfile.ZipFile(archive) as z:
            # Reject unexpected paths in third-party archives.
            for name in z.namelist():
                if not (DATA/name).resolve().is_relative_to(DATA.resolve()):raise ValueError('Unsafe archive path')
            z.extractall(DATA)
    corpus=read_rows(DATA/'scifact/corpus.jsonl'); queries={q['_id']:q['text'] for q in read_rows(DATA/'scifact/queries.jsonl')}
    qrels=defaultdict(set)
    with open(DATA/'scifact/qrels/test.tsv') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if int(r['score'])>0:qrels[r['query-id']].add(r['corpus-id'])
    papers=[{'id':d['_id'],'title':d['title'],'abstract':d['text'],'references':[]} for d in corpus]
    e=MiningEngine(papers);rows=[]
    for method in ['tfidf','bm25']:
        for qid,relevant in sorted(qrels.items()):
            t=time.perf_counter();scores=e.lexical_scores(queries[qid],method)
            order=sorted(range(len(e.ids)),key=lambda i:(-scores[i],e.ids[i]))
            ranked=[e.ids[i] for i in order if scores[i]>0]
            ms=(time.perf_counter()-t)*1000
            rows.append(dict(query_id=qid,method=method,latency_ms=ms,**ranking_metrics(ranked,relevant)))
        print('retrieval',method,flush=True)
    save_rows('retrieval_per_query.jsonl',rows)
    return {'dataset':'BEIR SciFact test: 5,183 scientific abstracts / 300 held-out claims','documents':len(corpus),'queries':len(qrels),
            'source':'https://github.com/beir-cellar/beir','archive_sha256':sha(archive),'qrels_sha256':sha(DATA/'scifact/qrels/test.tsv'),
            'protocol':'Unsupervised fixed BM25 k1=1.5 b=0.75 versus sublinear TF-IDF. Both index title twice + abstract. Full corpus candidates; AP over all positive-score retrieved documents, with unretrieved relevant documents contributing zero. No qrel-based tuning.',
            'summary':summarize(rows,['ndcg_at_10','map','precision_at_10','recall_at_10','mrr','latency_ms'])}

def recommendations(engine,n_tasks=120):
    rng=np.random.default_rng(42)
    eligible=sorted(p for p in engine.ids if engine.graph.out_degree(p)>=6)
    chosen=sorted(rng.choice(eligible,size=min(n_tasks,len(eligible)),replace=False).tolist())
    held=set();tasks=[]
    for p in chosen:
        refs=sorted(engine.graph.successors(p));test=set(rng.choice(refs,size=max(1,len(refs)//3),replace=False).tolist())
        held.update((p,r) for r in test);tasks.append({'paper_id':p,'seeds':[x for x in refs if x not in test],'relevant':sorted(test)})
    trainpapers=[dict(p,references=[r for r in p.get('references',[]) if (p['id'],r) not in held]) for p in engine.papers]
    train=MiningEngine(trainpapers)
    assert not any(train.graph.has_edge(a,b) for a,b in held)
    save_rows('recommendation_split.jsonl',tasks)
    rows=[];coverage=defaultdict(set)
    for ti,task in enumerate(tasks):
        seeds=task['seeds'];exclusions=set(seeds)|{task['paper_id']}
        # Candidate pool fixed across methods; source paper and seen references excluded.
        profile=np.asarray(train.tfidf[[train.pos[p] for p in seeds]].mean(axis=0));profile/=np.linalg.norm(profile) or 1
        content=np.asarray(train.tfidf @ profile.T).ravel()
        ppr=nx.pagerank(train.graph,alpha=.85,personalization={p:float(p in seeds)/len(seeds) for p in train.ids},tol=1e-9,max_iter=300)
        graph=np.array([ppr[p] for p in train.ids]);allowed=np.array([p not in exclusions for p in train.ids])
        c=content/(content[allowed].max(initial=0) or 1);g=graph/(graph[allowed].max(initial=0) or 1)
        allscores={'popular':unit(train.pr),'content':c,'graph':g,'hybrid':.7*c+.3*g,'hybrid_mmr':.7*c+.3*g}
        for method,scores in allscores.items():
            order=[i for i in sorted(range(len(train.ids)),key=lambda i:(-scores[i],train.ids[i])) if allowed[i]]
            if method=='hybrid_mmr':
                pool=order[:150];chosen_indices=[]
                while pool and len(chosen_indices)<10:
                    sim=np.asarray((train.tfidf[pool] @ train.tfidf[chosen_indices].T).max(axis=1).toarray()).ravel() if chosen_indices else np.zeros(len(pool))
                    at=int(np.argmax(.85*scores[pool]-.15*sim));chosen_indices.append(pool.pop(at))
                order=chosen_indices+[i for i in order if i not in chosen_indices]
            ranked=[train.ids[i] for i in order]; metrics=ranking_metrics(ranked,task['relevant']);top=order[:10]
            sim=(train.tfidf[top] @ train.tfidf[top].T).toarray();diversity=1-float(sim[np.triu_indices(len(top),1)].mean())
            rows.append(dict(paper_id=task['paper_id'],method=method,diversity=diversity,**metrics));coverage[method].update(ranked[:10])
        if (ti+1)%30==0:print('recommendations',ti+1,flush=True)
    save_rows('recommendation_per_paper.jsonl',rows)
    summary=summarize(rows,['recall_at_10','ndcg_at_10','precision_at_10','mrr','diversity'])
    for s in summary:s['catalog_coverage']=len(coverage[s['method']])/len(train.ids)
    diff=[]
    for t in tasks:
        r={r['method']:r['ndcg_at_10'] for r in rows if r['paper_id']==t['paper_id']}
        diff.append(r['hybrid']-r['content'])
    return {'tasks':len(tasks),'eligible_sources':len(eligible),'masked_edges':len(held),'train_edges':train.graph.number_of_edges(),
            'protocol':f'Random reference holdout: sample {len(tasks)} source papers with >=6 in-corpus references; hide floor(degree/3) links per source globally before all graph computations. Remaining references form seed profile. Source and seed papers excluded. All remaining corpus papers are candidates; no temporal generalization claim.',
            'split_sha256':sha(OUT/'recommendation_split.jsonl'),'summary':summary,'paired_hybrid_minus_content_ndcg':{'mean':float(np.mean(diff)),'ci95':bootstrap_mean(diff)},
            'limitations':'Citation recovery is a proxy for recommendation; no observed user interactions or satisfaction labels. Transductive content; held-out papers remain in vocabulary. Future graph structure is not time-censored.'}

def review_eval(reviews):
    rows=deduplicate(reviews);groups={split:{r['paper_id'] for r in rows if r['split']==split} for split in ['train','dev','test']}
    assert not(groups['train']&groups['test'] or groups['train']&groups['dev'] or groups['test']&groups['dev'])
    train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']=='test'];dev=[r for r in rows if r['split']=='dev']
    model=train_model(rows);y=[rating_label(r['rating']) for r in test]
    majority=Counter(rating_label(r['rating']) for r in train).most_common(1)[0][0]
    predictions={'majority':[majority]*len(test),'aspect_lexicon':[opinion(r['text'])['label'] for r in test],
                 'tfidf_logreg':model.predict([r['text'] for r in test]).tolist()}
    summary=[];per=[]
    for name,pred in predictions.items():
        # Cluster bootstrap preserves correlation between reviews of one paper.
        rng=np.random.default_rng(42);papers=sorted(groups['test']);by_p={p:[i for i,r in enumerate(test) if r['paper_id']==p] for p in papers};f1s=[]
        for _ in range(1000):
            sample=rng.choice(papers,len(papers),replace=True);idx=[i for p in sample for i in by_p[p]]
            f1s.append(f1_score(np.array(y)[idx],np.array(pred)[idx],labels=LABELS,average='macro',zero_division=0))
        summary.append({'method':name,'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0)),
            'macro_f1_ci95':[float(x) for x in np.quantile(f1s,[.025,.975])], 'confusion_matrix':confusion_matrix(y,pred,labels=LABELS).tolist(),
            'classification_report':classification_report(y,pred,labels=LABELS,output_dict=True,zero_division=0)})
        per.extend({'review_id':r['id'],'paper_id':r['paper_id'],'method':name,'gold_proxy':label,'prediction':p} for r,label,p in zip(test,y,pred))
    save_rows('review_predictions.jsonl',per)
    return {'dataset':'PeerRead ICLR 2017 scored reviews','split_sizes':{s:sum(r['split']==s for r in rows) for s in groups},
            'paper_counts':{s:len(v) for s,v in groups.items()},'test_label_counts':dict(Counter(y)), 'labels':LABELS,
            'protocol':'Official paper-disjoint train/dev/test split. Deduplicate normalized text within paper. Train only on train reviews; fixed C=1, no tuning on test. Rating <=4 critical, 5–6 mixed, >=7 positive. 95% cluster bootstrap intervals over test papers, 1000 replicates.',
            'dev_macro_f1':float(f1_score([rating_label(r['rating']) for r in dev],model.predict([r['text'] for r in dev]),labels=LABELS,average='macro',zero_division=0)),
            'summary':summary,'limitations':'Rating-derived polarity is weak supervision, not independent sentence sentiment annotations. Aspect rules have no human-labeled aspect benchmark.'}

def graph_eval(e):
    degree=np.array([e.graph.in_degree(p) for p in e.ids]);local=e.pr;external=np.array([p.get('citations',0) for p in e.papers])
    groups=defaultdict(set)
    for p,c in e.communities.items():groups[c].add(p)
    modularity=nx.community.modularity(e.graph.to_undirected(),list(groups.values()))
    return dict(e.stats(),pagerank_indegree_spearman=float(spearmanr(local,degree).statistic),pagerank_external_citations_spearman=float(spearmanr(local,external).statistic),
                modularity=float(modularity),direction='citing -> cited',damping=.85,community_algorithm='Louvain on undirected projection, seed 42')

def latency(e):
    queries=['graph neural networks','recommender systems','sentiment analysis','knowledge graphs','reinforcement learning','information retrieval','deep learning','large language models']
    result=[]
    for method in ['bm25','tfidf','hybrid']:
        for q in queries:e.search(q,method)
        elapsed=[]
        for _ in range(10):
            for q in queries:
                start=time.perf_counter();e.search(q,method);elapsed.append((time.perf_counter()-start)*1000)
        result.append({'method':method,'runs':len(elapsed),'p50_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95))})
    return {'scope':'Warm in-process full search incl filtering, ranking, top 20 serialization; excludes HTTP/browser. Eight queries x ten repetitions, single process.', 'summary':result}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tasks',type=int,default=120);args=ap.parse_args()
    if args.tasks<1:ap.error('--tasks must be positive')
    OUT.mkdir(exist_ok=True)
    papers=read_rows(DATA/'papers.jsonl');reviews=read_rows(DATA/'reviews.jsonl');e=MiningEngine(papers)
    result={'generated_at':datetime.now(timezone.utc).isoformat(),'seed':42,'environment':{'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'cpu_count':os.cpu_count(),'numpy':np.__version__,'sklearn':sklearn.__version__,'networkx':nx.__version__},
            'corpus_sha256':sha(DATA/'papers.jsonl'),'reviews_sha256':sha(DATA/'reviews.jsonl')}
    result['graph']=graph_eval(e);print('graph done',flush=True)
    result['retrieval']=retrieval();print('retrieval done',flush=True)
    result['recommendation']=recommendations(e,args.tasks);print('recommendation done',flush=True)
    result['opinion']=review_eval(reviews);print('opinion done',flush=True)
    result['latency']=latency(e)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print('Saved',OUT/'results.json',flush=True)
if __name__=='__main__':main()
