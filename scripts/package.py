"""Package the reproducible handoff without environments, cookies, databases or secrets."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables';OUT.mkdir(exist_ok=True)
root_files=['README.md','Makefile','pyproject.toml','uv.lock','requirements.lock','Dockerfile','compose.yaml','.gitignore','.dockerignore','.env.example','.python-version','ResearchLens_Visual_Spec.pdf','20261_syllabus.pdf','HUST_THEME_BEAMER.zip']
paths=[ROOT/f for f in root_files]
for folder in ['researchlens','scripts','tests','docs','data','report','slides','artifacts']:
 for p in (ROOT/folder).rglob('*'):
  if not p.is_file():continue
  if any(part in {'__pycache__','.pytest_cache','qa'} for part in p.parts):continue
  if 'conferences/raw' in p.as_posix() or '.sqlite' in p.name:continue
  if p.suffix in {'.db','.db-shm','.db-wal','.pyc','.aux','.out','.toc','.nav','.snm','.fls','.fdb_latexmk','.xdv','.bbl','.blg','.lof'}:continue
  if p.name.endswith(('.db-shm','.db-wal')):continue
  if folder in {'report','slides'} and (p.suffix=='.log' or p.name=='main.pdf'):continue
  paths.append(p)
study=ROOT/'evaluation/reading-study'
paths += [study/name for name in ('README.md','annotate.html','blind_packet.json','frozen_runs.json','assistant_pilot.json','pilot_results.json','laya_results.json','laya_save_results.json')]
decisions=ROOT/'evaluation/decision-study'
paths += [decisions/name for name in ('README.md','CSFCube_LICENSE.md','specter_method_ranked.json','features.jsonl','feature_provenance.json','reading_policy.json','text_policy.json','decision_comparison.json','passage_comparison.json','context_protocol.md','context_comparison.json','feedback_protocol.md','feedback_comparison.json','method_findings.md','method_retrieval_protocol.md','method_retrieval.json','method_decision_protocol.md','method_decision.json')]
paths += [OUT/'ResearchLens_Report.pdf',OUT/'ResearchLens_Slides.pdf']
archive=OUT/'ResearchLens_Source.zip'
with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
 for path in sorted(set(paths)):z.write(path,Path('ResearchLens')/path.relative_to(ROOT))
manifest={'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [OUT/'ResearchLens_Report.pdf',OUT/'ResearchLens_Slides.pdf',archive]},'source_files':len(set(paths))}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
