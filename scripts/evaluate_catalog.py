"""Measure actual conference coverage and warm FTS latency; no relevance claim."""
import hashlib,json,platform,sqlite3,tempfile,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from researchlens.catalog import Catalog,CONFERENCES
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts'
QUERIES=['graph neural networks','large language models','information retrieval','diffusion models','reinforcement learning','machine translation','object detection','semantic segmentation','speech recognition','causal inference','federated learning','domain adaptation','question answering','contrastive learning','adversarial robustness','knowledge graphs','image generation','vision transformers','sentiment analysis','recommendation systems']
def main():
    source=ROOT/'data/conferences/papers.jsonl';manifest=json.loads(source.with_name('manifest.json').read_text())
    sha=hashlib.sha256(source.read_bytes()).hexdigest();assert sha==manifest['sha256']
    with tempfile.TemporaryDirectory(prefix='researchlens-catalog-eval-') as tmp:
        catalog=Catalog(Path(tmp)/'catalog.sqlite');start=time.perf_counter();catalog.sync(source);build_s=time.perf_counter()-start
        stats=catalog.stats();assert stats['papers']==manifest['papers']
        measurements=[]
        for query in QUERIES:catalog.search(query)
        for repeat in range(5):
            for i,query in enumerate(QUERIES):
                conference=CONFERENCES[i%len(CONFERENCES)] if repeat%2 else None
                start=time.perf_counter();result=catalog.search(query,conference=conference,limit=20);elapsed=1000*(time.perf_counter()-start)
                measurements.append(dict(query=query,conference=conference,repeat=repeat,elapsed_ms=elapsed,total=result['total']))
        index_bytes=catalog.path.stat().st_size
    latency=[x['elapsed_ms'] for x in measurements]
    result=dict(generated_at=datetime.now(timezone.utc).isoformat(),source_sha256=sha,papers=stats['papers'],coverage=manifest['coverage'],abstracts=sum(x['abstracts'] for x in stats['coverage']),index_build_s=build_s,index_bytes=index_bytes,source_bytes=source.stat().st_size,
                latency=dict(queries=len(measurements),p50_ms=float(np.percentile(latency,50)),p95_ms=float(np.percentile(latency,95)),max_ms=max(latency)),measurements=measurements,
                environment=dict(python=platform.python_version(),sqlite=sqlite3.sqlite_version,platform=platform.platform(),cpu=platform.machine()),
                protocol='Single process, fresh disk index, 20 fixed topical queries warmed once then five rounds; alternate unfiltered and conference-filtered rounds. Full search includes count, BM25 ranking and top20 JSON decoding. Excludes HTTP/browser/network and startup. Descriptive coverage/latency only; no relevance labels for this expanded corpus.')
    (OUT/'catalog_results.json').write_text(json.dumps(result,indent=2))
    years=range(manifest['years'][0],manifest['years'][1]+1);counts={(x['conference'],x['year']):x['papers'] for x in manifest['coverage']}
    values=np.array([[counts[c,y] for y in years] for c in CONFERENCES])
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(11,5.7));ax.imshow(np.log1p(values),cmap='Reds',vmin=0,vmax=np.log1p(values.max())*1.1,aspect='auto')
    for row,c in enumerate(CONFERENCES):
        for col,y in enumerate(years):ax.text(col,row,f'{counts[c,y]:,}',ha='center',va='center',fontsize=12,color='white' if values[row,col]>1200 else '#172b3a')
    ax.set_xticks(range(len(years)),years);ax.set_yticks(range(len(CONFERENCES)),CONFERENCES);ax.tick_params(length=0,pad=10)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title(f'{stats["papers"]:,} collected records across eight conferences',loc='left',fontweight='bold',fontsize=16,pad=20)
    fig.text(.13,.015,'Fetched metadata after deduplication; not a census of accepted papers. Source: crawl manifest.',fontsize=10,color='#65737c')
    fig.tight_layout(rect=(0,.04,1,1))
    for suffix in ('pdf','png'):fig.savefig(OUT/'figures'/('conference_coverage.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    macros={'CatalogCount':f'{stats["papers"]:,}'.replace(',','.'),'CatalogLatencyMedian':f'{result["latency"]["p50_ms"]:.1f}','CatalogLatencyTail':f'{result["latency"]["p95_ms"]:.1f}','CatalogBuildSeconds':f'{build_s:.1f}','CatalogIndexMiB':f'{index_bytes/1024**2:.1f}','CatalogAbstractPercent':f'{100*result["abstracts"]/stats["papers"]:.1f}'}
    (ROOT/'report/generated/catalog_metrics.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in macros.items())+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('measurements','coverage')},indent=2))
if __name__=='__main__':main()
