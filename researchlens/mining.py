"""Sparse lexical retrieval, citation mining and transparent recommendations."""
from __future__ import annotations
from collections import Counter
import math
import re
import shlex
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize

TOKEN = re.compile(r'(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b')

def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]

def unit(values):
    x=np.asarray(values,dtype=float)
    m=x.max(initial=0)
    return x/m if m > 0 else np.zeros_like(x)

def query_parts(query):
    """Quoted phrases, -exclusions and OR terms. Unquoted terms use OR semantics."""
    try: parts=shlex.split(query)
    except ValueError as exc: raise ValueError('Close the quotation marks in your query.') from exc
    positive=[]; excluded=[]; phrases=[]
    for p in parts:
        if p.upper() in {'AND','OR'}: continue
        if p.startswith('-') and len(p)>1: excluded.append(p[1:].lower())
        else:
            positive.append(p)
            if ' ' in p: phrases.append(p.lower())
    return ' '.join(positive),excluded,phrases,any(part.upper() == 'AND' for part in parts)

def graph_metrics(graph):
    if not graph: return {},{},{},{}
    pagerank=nx.pagerank(graph,alpha=.85,max_iter=300,tol=1e-10)
    ids=list(graph); a=nx.to_scipy_sparse_array(graph,nodelist=ids,dtype=float,format='csr')
    # Nonnegative power iteration avoids arbitrary signs from singular vectors.
    authority=np.ones(len(ids))/math.sqrt(len(ids)); hub=authority.copy()
    for _ in range(200):
        h=np.asarray(a @ authority).ravel(); h=h/(np.linalg.norm(h) or 1)
        new=np.asarray(a.T @ h).ravel(); new=new/(np.linalg.norm(new) or 1)
        if np.linalg.norm(new-authority)<1e-9: authority=new; hub=h; break
        authority=new; hub=h
    authority=authority/(authority.sum() or 1); hub=hub/(hub.sum() or 1)
    undirected=graph.to_undirected()
    groups=nx.community.louvain_communities(undirected,seed=42) if graph.number_of_edges() else [{i} for i in ids]
    groups=sorted(groups,key=lambda c:(-len(c),min(c)))
    communities={pid:i for i,g in enumerate(groups) for pid in g}
    return pagerank,dict(zip(ids,authority)),dict(zip(ids,hub)),communities

class MiningEngine:
    def __init__(self,papers):
        if not papers: raise ValueError('At least one paper is required.')
        self.papers=papers; self.ids=[p['id'] for p in papers]; self.pos={p:i for i,p in enumerate(self.ids)}
        if len(self.pos)!=len(papers): raise ValueError('Paper IDs must be unique.')
        self.by_id={p['id']:p for p in papers}
        self.texts=[p['title']+' '+p.get('abstract','') for p in papers]
        weighted=[p['title']+' '+p['title']+' '+p.get('abstract','') for p in papers]
        self.vectorizer=CountVectorizer(analyzer=tokens,max_features=60000)
        if not any(tokens(text) for text in weighted): raise ValueError('Corpus must contain indexable words.')
        self.counts=self.vectorizer.fit_transform(weighted).tocsr()
        self.transformer=TfidfTransformer(sublinear_tf=True)
        self.tfidf=self.transformer.fit_transform(self.counts).tocsr()
        df=np.asarray((self.counts>0).sum(axis=0)).ravel(); n=len(papers)
        self.idf=np.log1p((n-df+.5)/(df+.5))
        self.length=np.asarray(self.counts.sum(axis=1)).ravel(); self.avg_length=self.length.mean() or 1
        self.graph=nx.DiGraph(); self.graph.add_nodes_from(self.ids)
        self.graph.add_edges_from((p['id'],r) for p in papers for r in p.get('references',[]) if r in self.pos and r!=p['id'])
        self.pagerank,self.authority,self.hubs,self.communities=graph_metrics(self.graph)
        self.pr=np.array([self.pagerank[p] for p in self.ids])
        self.features=self.vectorizer.get_feature_names_out()

    def lexical_scores(self,query,method='bm25'):
        q=self.vectorizer.transform([query]); columns=q.indices
        if not len(columns): return np.zeros(len(self.ids))
        if method=='tfidf': return np.asarray((self.tfidf @ self.transformer.transform(q).T).toarray()).ravel()
        tf=self.counts[:,columns].toarray(); norm=1.5*(1-.75+.75*self.length/self.avg_length)
        return (self.idf[columns]*tf*2.5/(tf+norm[:,None])).sum(axis=1)

    def search(self,query='',method='hybrid',year_min=None,year_max=None,topic=None,venue=None,open_access=False,offset=0,limit=20):
        if method not in {'bm25','tfidf','hybrid','pagerank'}: raise ValueError('Unknown ranking method')
        clean,excluded,phrases,and_mode=query_parts(query)
        lexical=self.lexical_scores(clean,'tfidf' if method=='tfidf' else 'bm25')
        score=lexical.copy()
        if method=='hybrid': score=.85*unit(lexical)+.15*unit(self.pr)
        if method=='pagerank' or not query.strip(): score=self.pr.copy()
        indices=[]
        wanted=set(tokens(clean))
        for i,p in enumerate(self.papers):
            text=self.texts[i].lower()
            if clean and (not wanted or lexical[i]<=0): continue
            def contains(term):
                pattern=r'(?<!\w)'+re.escape(term).replace(r'\ ',r'\s+')+r'(?!\w)'
                return re.search(pattern,text) is not None
            if any(not contains(x) for x in phrases) or any(contains(x) for x in excluded): continue
            if and_mode and not wanted.issubset(set(tokens(text))): continue
            year=p.get('year') or 0
            if year_min is not None and year<year_min: continue
            if year_max is not None and year>year_max: continue
            if topic and topic not in p.get('topics',[]) and topic!=p.get('collection_topic'): continue
            if venue and p.get('venue')!=venue: continue
            if open_access and not p.get('open_access'): continue
            indices.append(i)
        indices.sort(key=lambda i:(-score[i],self.ids[i]))
        return {'total':len(indices),'items':[self.summary(i,float(score[i]),float(lexical[i])) for i in indices[offset:offset+limit]]}

    def summary(self,i,score=None,lexical=None):
        p=self.papers[i]
        result={k:p.get(k) for k in ('id','title','abstract','authors','year','venue','topics','collection_topic','citations','source','url','open_access')}
        result.update(pagerank=float(self.pr[i]),authority=float(self.authority[p['id']]),community=self.communities[p['id']],local_citations=self.graph.in_degree(p['id']))
        if score is not None: result['score']=score
        if lexical is not None: result['lexical_score']=lexical
        return result

    def keywords(self,pid,n=8):
        row=self.tfidf.getrow(self.pos[pid]); order=np.argsort(-row.data)[:n]
        return [{'term':str(self.features[row.indices[i]]),'weight':round(float(row.data[i]),4)} for i in order]

    def _co_cited(self, seeds, limit):
        """Each citing paper casts at most one vote per candidate."""
        saved = set(seeds)
        witnesses = {}
        citing_papers = {source for seed in saved for source in self.graph.predecessors(seed)}
        for source in sorted(citing_papers):
            references = set(self.graph.successors(source))
            shared_seed = min(references & saved)
            for candidate in references - saved:
                witnesses.setdefault(candidate, []).append(
                    {"paper_id": source, "seed_id": shared_seed}
                )

        ranked = sorted(witnesses, key=lambda pid: (-len(witnesses[pid]), pid))
        items = []
        for pid in ranked[:limit]:
            evidence = witnesses[pid]
            count = len(evidence)
            reason = (f"{count} papers cite this paper alongside a saved paper." if count > 1
                      else "One paper cites this paper alongside a saved paper.")
            items.append(dict(self.summary(self.pos[pid], float(count)),
                              co_citation_count=count, witnesses=[dict(w, paper_title=self.by_id[w["paper_id"]]["title"],
                                  seed_title=self.by_id[w["seed_id"]]["title"]) for w in evidence[:3]], reason=reason))
        return {"cold_start": False, "items": items}

    def recommend(self,seeds,limit=12,method='hybrid',diversity=.15):
        seeds=list(dict.fromkeys(seeds))
        if method not in {'content','graph','hybrid','popular','cocitation'}: raise ValueError('Unknown recommendation method')
        if any(s not in self.pos for s in seeds): raise ValueError('Unknown seed paper')
        if not 0<=diversity<=1: raise ValueError('Diversity must be between zero and one')
        if not seeds:
            ordered=np.argsort(-self.pr,kind='stable')[:limit]
            return {'cold_start':True,'items':[dict(self.summary(i,float(self.pr[i])),reason='Cold start: influential within this corpus.') for i in ordered]}
        if method == "cocitation":
            return self._co_cited(seeds, limit)
        profile=normalize(np.asarray(self.tfidf[[self.pos[s] for s in seeds]].mean(axis=0)))
        content=np.asarray(self.tfidf @ profile.T).ravel()
        personalization={pid:1/len(seeds) if pid in seeds else 0 for pid in self.ids}
        pr=nx.pagerank(self.graph,alpha=.85,personalization=personalization,max_iter=300,tol=1e-9)
        graph=np.array([pr[p] for p in self.ids]); allowed=np.array([p not in seeds for p in self.ids])
        # Normalize on eligible candidates; seed mass must not flatten graph contribution.
        c=content/(content[allowed].max(initial=0) or 1); g=graph/(graph[allowed].max(initial=0) or 1)
        scores={'content':c,'graph':g,'hybrid':.7*c+.3*g,'popular':unit(self.pr)}[method]
        pool=[i for i in np.argsort(-scores,kind='stable') if allowed[i]][:max(150,limit)]
        selected=[]
        while pool and len(selected)<limit:
            redundancy=np.asarray((self.tfidf[pool] @ self.tfidf[selected].T).max(axis=1).toarray()).ravel() if selected else np.zeros(len(pool))
            mmr=(1-diversity)*scores[pool]-diversity*redundancy
            j=int(np.argmax(mmr)); selected.append(pool.pop(j))
        items=[]
        for i in selected:
            shared=set(tokens(self.texts[i])) & set(tokens(' '.join(self.texts[self.pos[s]] for s in seeds)))
            words=[x['term'] for x in self.keywords(self.ids[i],20) if x['term'] in shared][:4]
            explanations = {
                "content": f"Normalized content similarity {c[i]:.3f}.",
                "popular": f"Global PageRank {self.pr[i]:.5f}.",
                "graph": f"Normalized personalized PageRank {g[i]:.3f}.",
                "hybrid": f"Content 0.7 × {c[i]:.3f} + personalized PageRank 0.3 × {g[i]:.3f}.",
            }
            reason = explanations[method]
            if words and method in {"content", "hybrid"}:
                reason += " Shared terms: " + ", ".join(words) + "."
            if diversity:
                reason += f" Diversity reranking enabled (weight {diversity:g})."
            items.append(dict(self.summary(i,float(scores[i])),content_score=float(c[i]),graph_score=float(g[i]),shared_terms=words,reason=reason))
        return {'cold_start':False,'items':items}

    def subgraph(self,pid=None,limit=70):
        if pid:
            if pid not in self.pos: raise KeyError(pid)
            neighborhood=set(self.graph.predecessors(pid)) | set(self.graph.successors(pid))
            ids=[pid]+sorted(neighborhood,key=lambda p:-self.pagerank[p])[:limit-1]
        else: ids=sorted(self.ids,key=lambda p:-self.pagerank[p])[:limit]
        sub=self.graph.subgraph(ids); layout=nx.spring_layout(sub.to_undirected(),seed=42,iterations=100,k=.75)
        return {'nodes':[{'id':p,'title':self.by_id[p]['title'],'year':self.by_id[p].get('year'),'pagerank':self.pagerank[p],
                         'community':self.communities[p],'x':float(layout[p][0]),'y':float(layout[p][1])} for p in ids],
                'edges':[{'source':a,'target':b} for a,b in sub.edges], 'direction':'citing → cited',
                'total_nodes':len(self.graph),'total_edges':self.graph.number_of_edges()}

    def stats(self):
        connected=[c for c in nx.weakly_connected_components(self.graph)]
        return {'papers':len(self.ids),'edges':self.graph.number_of_edges(),'communities':len(set(self.communities.values())),
            'isolates':len(list(nx.isolates(self.graph))),'largest_component':max(map(len,connected),default=0),
            'abstracts':sum(bool(p.get('abstract')) for p in self.papers),'sources':dict(Counter(p['source'] for p in self.papers)),
            'years':dict(sorted(Counter(p.get('year') for p in self.papers if p.get('year')).items())),
            'topics':dict(Counter(p['collection_topic'] for p in self.papers)),
            'top_papers':[self.summary(self.pos[p]) for p in sorted(self.ids,key=lambda p:-self.pagerank[p])[:10]],
            'community_sizes':dict(Counter(self.communities.values()))}

ASPECTS={'novelty':r'novel|original|contribut|incremental', 'soundness':r'theor|proof|correct|assumption|rigor|sound',
         'evaluation':r'experiment|baseline|dataset|benchmark|evaluation|ablation', 'clarity':r'clear|clarity|writing|written|readab|explain',
         'reproducibility':r'reproduc|code|implement|hyperparam|detail'}
POS={'good','excellent','strong','novel','clear','interesting','convincing','thorough','significant','impressive','useful','well'}
NEG={'weak','unclear','limited','missing','poor','incorrect','confusing','insufficient','flaw','problem','lack','lacks'}

def opinion(text):
    """Rule baseline with local negation and exact sentence evidence; not a factual verdict."""
    sentences=re.split(r'(?<=[.!?])\s+|\n+',text.strip())
    findings=[]
    for sentence in sentences:
        if not sentence: continue
        words=re.findall(r"[a-z]+",sentence.lower()); score=0
        for i,w in enumerate(words):
            s=1 if w in POS else -1 if w in NEG else 0
            if any(x in {'no','not','never','hardly','isn','doesn','wasn'} for x in words[max(0,i-3):i]): s=-s
            score+=s
        aspects=[a for a,pattern in ASPECTS.items() if re.search(pattern,sentence,re.I)]
        if aspects: findings.append({'text':sentence,'aspects':aspects,'polarity':'positive' if score>0 else 'critical' if score<0 else 'mixed','score':score})
    total=sum(x['score'] for x in findings)
    return {'label':'positive' if total>0 else 'critical' if total<0 else 'mixed','score':total,'evidence':findings,
            'method':'Aspect keywords + polarity lexicon with three-token negation; sentence-level heuristic.'}

def extract(text):
    patterns={'datasets':r'\b(?:ImageNet|CIFAR[- ]?(?:10|100)|MNIST|SVHN|SQuAD|GLUE|Cora|CiteSeer|PubMed|MovieLens|COCO|WikiText|MS MARCO)\b',
        'methods':r'\b(?:BERT|GPT[- ]?\d*|Transformer[s]?|GCN|GAT|GraphSAGE|LSTM|CNN|SVM|PageRank|BM25|TF[- ]?IDF|reinforcement learning|contrastive learning)\b',
        'doi':r'10\.\d{4,9}/[^\s<>"\]]+', 'arxiv':r'\b\d{4}\.\d{4,5}(?:v\d+)?\b'}
    return {kind:[{'text':m.group(),'start':m.start(),'end':m.end(),'method':'pattern'} for m in re.finditer(pattern,text,re.I)] for kind,pattern in patterns.items()}
