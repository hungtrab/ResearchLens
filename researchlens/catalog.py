"""Disk-backed conference catalog; its crawl does not change the benchmark snapshot."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from contextlib import contextmanager

CONFERENCES=('ICML','NeurIPS','ICLR','AAAI','EMNLP','NAACL','ACL','CVPR')

def title_key(title):
    return re.sub(r'\W+', '', title.casefold())

class Catalog:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS catalog(id TEXT PRIMARY KEY,title TEXT,abstract TEXT,authors TEXT,conference TEXT,year INTEGER,citations INTEGER,payload TEXT);
            CREATE INDEX IF NOT EXISTS catalog_year ON catalog(conference,year);
            CREATE VIRTUAL TABLE IF NOT EXISTS catalog_fts USING fts5(title,abstract,authors,conference,content='catalog',content_rowid='rowid',tokenize='unicode61');
            CREATE TABLE IF NOT EXISTS catalog_meta(key TEXT PRIMARY KEY,value TEXT);
            ''')
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=60)
        try:
            yield db;db.commit()
        except Exception:
            db.rollback();raise
        finally:db.close()
    def sync(self,source):
        source=Path(source)
        if not source.exists():return
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        with self.connect() as db:
            if db.execute("SELECT value FROM catalog_meta WHERE key='sha256'").fetchone()==(digest,):return
            db.execute('DELETE FROM catalog')
            rows=[]
            with source.open() as f:
                for line in f:
                    if not line.strip():continue
                    p=json.loads(line);rows.append((p['id'],p['title'],p.get('abstract',''),' '.join(p.get('authors',[])),p['conference'],p['year'],p.get('citations',0),json.dumps(p,ensure_ascii=False)))
                    if len(rows)>=1000:db.executemany('INSERT INTO catalog VALUES(?,?,?,?,?,?,?,?)',rows);rows=[]
            db.executemany('INSERT INTO catalog VALUES(?,?,?,?,?,?,?,?)',rows)
            db.execute("INSERT INTO catalog_fts(catalog_fts) VALUES('rebuild')")
            db.execute("INSERT OR REPLACE INTO catalog_meta VALUES('sha256',?)",(digest,))
    def get(self,pid):
        with self.connect() as db:row=db.execute('SELECT payload FROM catalog WHERE id=?',(pid,)).fetchone()
        return json.loads(row[0]) if row else None
    def search(self,q='',conference=None,year_min=None,year_max=None,offset=0,limit=20):
        terms=list(dict.fromkeys(re.findall(r'[^\W_]+',q,flags=re.UNICODE)))[:32]
        clauses=[];params=[]
        join='';rank='c.year DESC,c.citations DESC,c.title,c.id'
        if terms:
            join=' JOIN catalog_fts ON catalog_fts.rowid=c.rowid';clauses.append('catalog_fts MATCH ?')
            params.append(' OR '.join('"'+term.replace('"','""')+'"' for term in terms));rank='bm25(catalog_fts,8,2,2,1),c.year DESC,c.id'
        elif q.strip():return {'total':0,'items':[],'method':'SQLite FTS5 BM25'}
        if conference:clauses.append('c.conference=?');params.append(conference)
        if year_min is not None:clauses.append('c.year>=?');params.append(year_min)
        if year_max is not None:clauses.append('c.year<=?');params.append(year_max)
        where=' WHERE '+' AND '.join(clauses) if clauses else ''
        with self.connect() as db:
            count=db.execute('SELECT COUNT(*) FROM catalog c'+join+where,params).fetchone()[0]
            rows=db.execute('SELECT c.payload FROM catalog c'+join+where+' ORDER BY '+rank+' LIMIT ? OFFSET ?',params+[limit,offset]).fetchall()
        return {'total':count,'items':[json.loads(r[0]) for r in rows],'method':'SQLite FTS5 BM25'}
    def stats(self):
        with self.connect() as db:
            count=db.execute('SELECT COUNT(*) FROM catalog').fetchone()[0]
            rows=db.execute('SELECT conference,year,COUNT(*),SUM(abstract!=\'\') FROM catalog GROUP BY conference,year ORDER BY conference,year').fetchall()
        return {'papers':count,'coverage':[dict(conference=c,year=y,papers=n,abstracts=a) for c,y,n,a in rows]}
