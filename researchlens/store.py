"""SQLite persistence. Profiles are local anonymous browser sessions."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

class Store:
    def __init__(self,path):
        self.path=str(path); Path(path).parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS papers(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS saved(profile TEXT NOT NULL,paper_id TEXT NOT NULL REFERENCES papers(id),created TEXT DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(profile,paper_id));
            CREATE TABLE IF NOT EXISTS history(profile TEXT NOT NULL,query TEXT NOT NULL,created TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY,profile TEXT NOT NULL,title TEXT NOT NULL,brief TEXT NOT NULL,updated TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS project_papers(project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,paper_id TEXT NOT NULL REFERENCES papers(id),created TEXT DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(project_id,paper_id));
            CREATE TABLE IF NOT EXISTS reading(profile TEXT NOT NULL,paper_id TEXT NOT NULL REFERENCES papers(id),status TEXT NOT NULL DEFAULT 'queued',page INTEGER NOT NULL DEFAULT 1,notes TEXT NOT NULL DEFAULT '',updated TEXT DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(profile,paper_id));
            CREATE TABLE IF NOT EXISTS imported(paper_id TEXT PRIMARY KEY REFERENCES papers(id),created TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS import_aliases(external_id TEXT PRIMARY KEY,paper_id TEXT NOT NULL REFERENCES papers(id));
            CREATE INDEX IF NOT EXISTS projects_profile ON projects(profile);
            ''')
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=20); db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback(); raise
        finally: db.close()
    def ingest(self,papers):
        with self.connect() as db:
            db.executemany('INSERT INTO papers(id,payload) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',[(p['id'],json.dumps(p)) for p in papers])
    def saved(self,profile):
        with self.connect() as db: return [r[0] for r in db.execute('SELECT paper_id FROM saved WHERE profile=? ORDER BY created,paper_id',(profile,))]
    def save(self,profile,pid):
        with self.connect() as db: db.execute('INSERT OR IGNORE INTO saved(profile,paper_id) VALUES (?,?)',(profile,pid))
    def remove(self,profile,pid):
        with self.connect() as db: db.execute('DELETE FROM saved WHERE profile=? AND paper_id=?',(profile,pid))
    def remember(self,profile,query):
        with self.connect() as db:
            db.execute('INSERT INTO history(profile,query) VALUES (?,?)',(profile,query))
            db.execute('DELETE FROM history WHERE profile=? AND rowid NOT IN (SELECT rowid FROM history WHERE profile=? ORDER BY rowid DESC LIMIT 20)',(profile,profile))
    def history(self,profile):
        with self.connect() as db: return [r[0] for r in db.execute('SELECT query FROM history WHERE profile=? ORDER BY rowid DESC LIMIT 8',(profile,))]
    def project(self,profile,pid):
        with self.connect() as db:
            row=db.execute('SELECT id,title,brief,updated FROM projects WHERE profile=? AND id=?',(profile,pid)).fetchone()
            if row is None: return None
            result=dict(zip(['id','title','brief','updated'],row))
            result['paper_ids']=[r[0] for r in db.execute('SELECT paper_id FROM project_papers WHERE project_id=? ORDER BY created,paper_id',(pid,))]
            return result
    def projects(self,profile):
        with self.connect() as db: ids=[r[0] for r in db.execute('SELECT id FROM projects WHERE profile=? ORDER BY updated DESC,id',(profile,))]
        return [self.project(profile,pid) for pid in ids]
    def write_project(self,profile,pid,title,brief):
        with self.connect() as db:
            db.execute('INSERT INTO projects(id,profile,title,brief) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET title=excluded.title,brief=excluded.brief,updated=CURRENT_TIMESTAMP WHERE projects.profile=excluded.profile',(pid,profile,title,brief))
    def delete_project(self,profile,pid):
        with self.connect() as db: db.execute('DELETE FROM projects WHERE id=? AND profile=?',(pid,profile))
    def project_paper(self,profile,project_id,pid,add):
        with self.connect() as db:
            if not db.execute('SELECT 1 FROM projects WHERE id=? AND profile=?',(project_id,profile)).fetchone(): raise KeyError(project_id)
            if add: db.execute('INSERT OR IGNORE INTO project_papers(project_id,paper_id) VALUES (?,?)',(project_id,pid))
            else: db.execute('DELETE FROM project_papers WHERE project_id=? AND paper_id=?',(project_id,pid))
            db.execute('UPDATE projects SET updated=CURRENT_TIMESTAMP WHERE id=?',(project_id,))
    def reading(self,profile,pid):
        with self.connect() as db:
            row=db.execute('SELECT status,page,notes,updated FROM reading WHERE profile=? AND paper_id=?',(profile,pid)).fetchone()
        return dict(zip(['status','page','notes','updated'],row)) if row else {'status':'queued','page':1,'notes':'','updated':None}
    def write_reading(self,profile,pid,value):
        with self.connect() as db:
            db.execute('INSERT INTO reading(profile,paper_id,status,page,notes) VALUES (?,?,?,?,?) ON CONFLICT(profile,paper_id) DO UPDATE SET status=excluded.status,page=excluded.page,notes=excluded.notes,updated=CURRENT_TIMESTAMP',(profile,pid,value['status'],value['page'],value['notes']))
    def readings(self,profile):
        with self.connect() as db:
            rows=db.execute('SELECT paper_id,status,page,notes,updated FROM reading WHERE profile=? ORDER BY updated DESC,paper_id',(profile,)).fetchall()
        return [dict(zip(['paper_id','status','page','notes','updated'],r)) for r in rows]
    def imported(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT p.payload FROM papers p JOIN imported i ON i.paper_id=p.id ORDER BY i.created,p.id')]
    def imported_alias(self,pid):
        with self.connect() as db:row=db.execute('SELECT paper_id FROM import_aliases WHERE external_id=?',(pid,)).fetchone()
        return row[0] if row else None
    def persist_import(self,paper,aliases):
        with self.connect() as db:
            db.execute('INSERT INTO papers(id,payload) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',(paper['id'],json.dumps(paper)))
            db.execute('INSERT OR IGNORE INTO imported(paper_id) VALUES(?)',(paper['id'],))
            db.executemany('INSERT OR REPLACE INTO import_aliases(external_id,paper_id) VALUES(?,?)',[(alias,paper['id']) for alias in aliases])
    def remember_alias(self,pid,alias):
        with self.connect() as db:db.execute('INSERT OR REPLACE INTO import_aliases(external_id,paper_id) VALUES(?,?)',(alias,pid))
    def aliases(self):
        with self.connect() as db:return dict(db.execute('SELECT external_id,paper_id FROM import_aliases'))
    def remember_aliases(self,pid,aliases):
        with self.connect() as db:db.executemany('INSERT OR REPLACE INTO import_aliases(external_id,paper_id) VALUES(?,?)',[(alias,pid) for alias in aliases])
