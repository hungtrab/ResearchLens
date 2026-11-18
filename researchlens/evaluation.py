"""Evaluation primitives. Binary relevance, full-ranking AP, deterministic bootstrap."""
import numpy as np

def ranking_metrics(ranked,relevant,k=10):
    relevant=set(relevant)
    if not relevant: raise ValueError('At least one relevant document is required')
    if len(ranked)!=len(set(ranked)): raise ValueError('Ranking must not contain duplicate IDs')
    hits=np.array([x in relevant for x in ranked],dtype=float)
    top=hits[:k]; discounts=1/np.log2(np.arange(2,len(top)+2))
    ideal=np.sum(1/np.log2(np.arange(2,min(k,len(relevant))+2)))
    at=np.flatnonzero(hits)
    return {'precision_at_10':float(top.sum()/k),'recall_at_10':float(top.sum()/len(relevant)),
            'ndcg_at_10':float((top*discounts).sum()/ideal),
            'map':float(sum(hits.cumsum()[i]/(i+1) for i in at)/len(relevant)),
            'mrr':float(1/(at[0]+1)) if len(at) else 0.0}

def bootstrap_mean(values,seed=42,samples=2000):
    a=np.asarray(values,dtype=float)
    if not len(a): raise ValueError('No observations')
    rng=np.random.default_rng(seed)
    means=a[rng.integers(0,len(a),(samples,len(a)))].mean(axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]

def summarize(rows,metrics):
    result=[]
    for method in dict.fromkeys(r['method'] for r in rows):
        subset=[r for r in rows if r['method']==method]
        summary={'method':method,'n':len(subset)}
        for metric in metrics:
            vals=[r[metric] for r in subset]; summary[metric]=float(np.mean(vals)); summary[metric+'_ci95']=bootstrap_mean(vals)
        result.append(summary)
    return result
