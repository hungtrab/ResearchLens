"""Generate report-ready vector charts exclusively from recorded measurements."""
import json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
import networkx as nx
from researchlens.mining import MiningEngine
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/figures';OUT.mkdir(exist_ok=True,parents=True)
r=json.load(open(ROOT/'artifacts/results.json'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#263e4c','text.color':'#172b3a','xtick.color':'#52636c','ytick.color':'#52636c','axes.edgecolor':'#b8c5cb','savefig.facecolor':'white','pdf.fonttype':42})
RED='#bd2636';BLUE='#476e80';GRAY='#aebcc4';TEAL='#37887e'
def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.15)
    fig.savefig(OUT/f'{name}.png',dpi=200,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)
def compare(name,rows,metric,labels,title,ylabel):
    fig,ax=plt.subplots(figsize=(9,4.5));values=[s[metric] for s in rows];ci=np.array([s[metric+'_ci95'] for s in rows]);x=np.arange(len(rows));colors=[RED if v==max(values) else BLUE for v in values]
    ax.bar(x,values,color=colors,width=.58);ax.errorbar(x,values,yerr=np.vstack([values-ci[:,0],ci[:,1]-values]),fmt='none',ecolor='#263e4c',capsize=5,lw=1.2)
    for i,v in enumerate(values):ax.text(i,ci[i,1]+.014,f'{v:.3f}',ha='center',fontsize=12,fontweight='bold')
    ax.set_xticks(x,labels);ax.set_ylim(0,max(ci[:,1])*1.22+.025);ax.set_ylabel(ylabel);ax.set_title(title,loc='left',fontweight='bold',pad=20);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    ax.text(0,-.22,'Error bars: 95% bootstrap intervals. Source: artifacts/results.json',transform=ax.transAxes,fontsize=9,color='#65737c')
    fig.tight_layout();save(fig,name)
compare('retrieval',r['retrieval']['summary'],'ndcg_at_10',['TF-IDF','BM25'],'BM25 ranks relevant scientific evidence higher','nDCG@10')
compare('recommendation',r['recommendation']['summary'],'ndcg_at_10',['Popularity','Content','Personalized\nPageRank','Hybrid','Hybrid\n+ MMR'],'Citation proximity leads on hidden-reference recovery','nDCG@10')
compare('opinion',r['opinion']['summary'],'macro_f1',['Majority','Aspect lexicon','TF-IDF +\nlogistic regression'],'Review polarity remains a difficult prediction task','Macro-F1')
fig,ax=plt.subplots(figsize=(8,4.7));m=np.array(r['opinion']['summary'][-1]['confusion_matrix']);im=ax.imshow(m,cmap='Blues',vmin=0);labels=r['opinion']['labels'];ax.set_xticks(range(3),labels);ax.set_yticks(range(3),labels);ax.set_xlabel('Predicted rating-derived polarity');ax.set_ylabel('Observed rating-derived polarity');ax.set_title('Confusion concentrates around mixed judgments',loc='left',pad=18,fontweight='bold')
for i in range(3):
 for j in range(3):ax.text(j,i,str(m[i,j]),ha='center',va='center',fontsize=20,color='white' if m[i,j]>m.max()/2 else '#172b3a')
fig.colorbar(im,ax=ax,label='Test reviews');fig.tight_layout();save(fig,'confusion')
fig,ax=plt.subplots(figsize=(8.5,4.6))
for s in r['recommendation']['summary']:
 ax.scatter(s['diversity'],s['ndcg_at_10'],s=1800*s['catalog_coverage']+60,color=RED if s['method']=='graph' else BLUE,edgecolors='white',zorder=3)
 ax.annotate({'popular':'Popularity','content':'Content','graph':'PPR','hybrid':'Hybrid','hybrid_mmr':'Hybrid + MMR'}[s['method']],(s['diversity'],s['ndcg_at_10']),xytext=(9,5),textcoords='offset points',fontsize=11)
ax.set(xlabel='Intra-list diversity (1 − mean cosine)',ylabel='nDCG@10',xlim=(.25,1.12),ylim=(0,.22));ax.set_title('MMR improves diversity and coverage of the hybrid',loc='left',pad=18,fontweight='bold');ax.grid(alpha=.13);ax.text(0,-.21,'Bubble area reflects catalog coverage. Diversity is a text-similarity proxy.',transform=ax.transAxes,fontsize=9);fig.tight_layout();save(fig,'diversity')
fig,axs=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={'width_ratios':[1.3,1]});topics=r['graph']['topics'];shorts={'graph neural networks':'Graph neural networks','information retrieval':'Information retrieval','recommender systems':'Recommender systems','sentiment analysis':'Sentiment analysis','large language models':'Large language models','deep learning':'Deep learning','reinforcement learning':'Reinforcement learning','knowledge graphs':'Knowledge graphs','peer review':'PeerRead additions'}
axs[0].barh(list(range(len(topics))),list(topics.values()),color=BLUE);axs[0].set_yticks(range(len(topics)),[shorts[t] for t in topics],fontsize=10);axs[0].invert_yaxis();axs[0].set_xlabel('Unique papers, assigned to first collection query')
years={int(y):v for y,v in r['graph']['years'].items() if int(y)>=2000};axs[1].bar(years.keys(),years.values(),color=BLUE,width=.8);axs[1].set_xlabel('Publication year');axs[1].set_ylabel('Papers');fig.suptitle('A focused collection, not a representative publication census',x=.05,ha='left',fontweight='bold',fontsize=14);fig.tight_layout();save(fig,'corpus')
fig,ax=plt.subplots(figsize=(8.5,4));rows=r['latency']['summary'];x=np.arange(len(rows));ax.bar(x-.16,[s['p50_ms'] for s in rows],width=.3,label='p50',color=BLUE);ax.bar(x+.16,[s['p95_ms'] for s in rows],width=.3,label='p95',color=RED);ax.set_xticks(x,[s['method'].upper() for s in rows]);ax.set_ylabel('Milliseconds');ax.set_title('Warm local search is measured in milliseconds',loc='left',fontweight='bold',pad=18);ax.legend(frameon=False);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True);ax.text(0,-.19,'80 queries/method; filtering + ranking; excludes HTTP and browser. CPU run.',transform=ax.transAxes,fontsize=9);fig.tight_layout();save(fig,'latency')
papers=[json.loads(x) for x in open(ROOT/'data/papers.jsonl',encoding='utf8') if x.strip()];e=MiningEngine(papers);ids=sorted(e.ids,key=lambda p:-e.pagerank[p])[:160];g=e.graph.subgraph(ids);pos=nx.spring_layout(g.to_undirected(),seed=42,iterations=180,k=.65)
fig,ax=plt.subplots(figsize=(10,6));palette=['#476e80','#bd5260','#76a08b','#baa16c','#7c88a6','#937888'];nx.draw_networkx_edges(g,pos,ax=ax,width=.65,alpha=.18,arrows=True,arrowsize=5,node_size=50);nx.draw_networkx_nodes(g,pos,ax=ax,node_size=[50+750*e.pagerank[p]/max(e.pr) for p in g],node_color=[palette[e.communities[p]%6] for p in g],edgecolors='white',linewidths=.8)
for p in ids[:5]:
 title=e.by_id[p]['title'];ax.annotate(title[:42]+('…' if len(title)>42 else ''),pos[p],xytext=(9,9),textcoords='offset points',fontsize=9,bbox=dict(fc='white',ec='none',alpha=.85))
ax.axis('off');ax.set_title('Citation structure of the 160 highest-PageRank papers',loc='left',fontweight='bold');fig.text(.13,.02,'Real observed edges · Node area: local PageRank · Colors: Louvain communities',fontsize=10,color='#65737c');save(fig,'network')
fig,ax=plt.subplots(figsize=(12,4.8));ax.set_xlim(0,12);ax.set_ylim(0,5);ax.axis('off')
def box(x,y,w,h,title,body,accent=False):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.08',fc='#f2f5f6',ec=RED if accent else '#c4d0d5',lw=1.3))
 ax.text(x+.15,y+h-.35,title,fontsize=12,fontweight='bold',color=RED if accent else '#172b3a');ax.text(x+.15,y+h-.7,body,fontsize=10,va='top',linespacing=1.6)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#71858f',lw=1.4))
box(.15,2.5,2.3,1.5,'PUBLIC SOURCES','OpenAlex works\nPeerRead scored reviews');box(3,2.5,2.45,1.5,'INGEST + CLEAN','JSONL cache + SHA-256\nDeduplication + provenance');box(6.05,2.5,2.5,1.5,'MINING CORE','Sparse retrieval + graphs\nReview model + ranking',True);box(9.15,2.5,2.6,1.5,'WORKSPACE','FastAPI + browser UI\nSQLite library + BibTeX');arrow((2.45,3.25),(3,3.25));arrow((5.45,3.25),(6.05,3.25));arrow((8.55,3.25),(9.15,3.25));box(3,.2,2.45,1.3,'APACHE SOLR','Optional Lucene BM25\nDocker profile');box(6.05,.2,2.5,1.3,'EXPERIMENTS','SciFact + reference holdout\nPaper-disjoint review test');arrow((4.2,2.5),(4.2,1.5));arrow((5.45,.85),(6.05,2.5));arrow((7.3,2.5),(7.3,1.5));box(9.15,.2,2.6,1.3,'ARTIFACTS','Per-query scores + intervals\nCharts + report + slides');arrow((8.55,.85),(9.15,.85));save(fig,'architecture')
# Dynamic TeX fragments prevent transcription drift in the documents.
def tex(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def tab(name,rows,keys):
 text='\n'.join(' & '.join(tex(row[k]) if isinstance(row[k],str) else f'{row[k]:.3f}' for k in keys)+r' \\' for row in rows)
 (OUT/f'{name}.tex').write_text(text+'\n')
tab('retrieval_table',r['retrieval']['summary'],['method','ndcg_at_10','map','precision_at_10','recall_at_10'])
tab('recommendation_table',r['recommendation']['summary'],['method','recall_at_10','ndcg_at_10','mrr','diversity','catalog_coverage'])
tab('opinion_table',r['opinion']['summary'],['method','accuracy','macro_f1'])
tab('latency_table',r['latency']['summary'],['method','p50_ms','p95_ms'])
print('Saved figures and tables:',OUT)
# Larger chart labels for projection; action titles are supplied by Beamer.
plt.rcParams.update({'font.size':16,'axes.labelsize':17,'xtick.labelsize':15,'ytick.labelsize':15})
for name,rows,metric,labels in [
 ('recommendation',r['recommendation']['summary'],'ndcg_at_10',['Popular','Content','PPR','Hybrid','Hybrid\n+ MMR']),
 ('retrieval',r['retrieval']['summary'],'ndcg_at_10',['TF-IDF','BM25']),
 ('opinion',r['opinion']['summary'],'macro_f1',['Majority','Lexicon','Logistic\nregression'])]:
 fig,ax=plt.subplots(figsize=(7.3,4.5));values=[s[metric] for s in rows];ci=np.array([s[metric+'_ci95'] for s in rows]);x=np.arange(len(rows))
 ax.bar(x,values,width=.6,color=[RED if v==max(values) else BLUE for v in values]);ax.errorbar(x,values,yerr=np.vstack([values-ci[:,0],ci[:,1]-values]),fmt='none',capsize=5,color='#172b3a',lw=1.5)
 for i,v in enumerate(values):ax.text(i,ci[i,1]+.012,f'{v:.3f}',ha='center',fontsize=16,fontweight='bold')
 ax.set_xticks(x,labels);ax.set_ylim(0,max(ci[:,1])*1.23+.015);ax.set_ylabel('Macro-F1' if metric=='macro_f1' else 'nDCG@10');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();save(fig,'slide_'+name)
fig,ax=plt.subplots(figsize=(7.3,4.5))
for s in r['recommendation']['summary']:
 ax.scatter(s['diversity'],s['ndcg_at_10'],s=2400*s['catalog_coverage']+80,color=RED if s['method']=='graph' else BLUE,zorder=3,edgecolors='white')
 ax.annotate({'popular':'Popular','content':'Content','graph':'PPR','hybrid':'Hybrid','hybrid_mmr':'Hybrid + MMR'}[s['method']],(s['diversity'],s['ndcg_at_10']),xytext=(-8,12) if s['method']=='hybrid_mmr' else (7,-16),textcoords='offset points',fontsize=14)
ax.set(xlim=(.26,1.14),ylim=(.01,.22),xlabel='Intra-list diversity',ylabel='nDCG@10');ax.grid(alpha=.15);fig.tight_layout();save(fig,'slide_diversity')
fig,ax=plt.subplots(figsize=(7.3,4.5));rows=r['latency']['summary'];x=np.arange(len(rows));ax.bar(x-.17,[s['p50_ms'] for s in rows],width=.3,color=BLUE,label='p50');ax.bar(x+.17,[s['p95_ms'] for s in rows],width=.3,color=RED,label='p95');ax.set_xticks(x,[s['method'].upper() for s in rows]);ax.set_ylabel('Milliseconds');ax.legend(frameon=False,ncol=2,loc='upper left');ax.set_ylim(0,max(s['p95_ms'] for s in rows)*1.35);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True);fig.tight_layout();save(fig,'slide_latency')
