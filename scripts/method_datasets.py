"""Verified source adapters for the method-retrieval study. No model sees judgments."""

import hashlib
import json
import re
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from researchlens.catalog import title_key

ARCHIVES = {
    "scierc.tar.gz": (
        "https://nlp.cs.washington.edu/sciIE/data/sciERC_raw.tar.gz",
        "15863ae2bfecd68bd9da412048bbbe52886259644bb6afcb46129b86dca12750",
    ),
    "arc.tar.gz": (
        "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/acl-arc.tar.gz",
        "90fa327317fa931cc49120bda64c6cf221eb2fdf80a0e5ef6c1a1869b96f4ebf",
    ),
}


def archive(cache, name):
    url, expected = ARCHIVES[name]
    path = cache / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Unexpected download {name}")
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Unexpected archive {name}")
    return tarfile.open(path)


@dataclass
class Query:
    key: str
    cluster: str
    text: str
    excluded: set
    grades: dict


def scierc(cache):
    with archive(cache, "scierc.tar.gz") as f:
        files = {m.name: f.extractfile(m).read() for m in f if m.isfile()}
    docs, mentions, postings = {}, defaultdict(list), defaultdict(set)
    audit = Counter()
    for name in sorted(k for k in files if k.endswith(".txt")):
        key = Path(name).stem
        doc = docs[key] = files[name].decode()
        tree = ET.fromstring(files[name + ".xml"])
        spans = []
        for sentence in tree.findall("./document/sentences/sentence"):
            tokens = sentence.findall("./tokens/token")
            spans.append(
                (
                    int(tokens[0].findtext("CharacterOffsetBegin")),
                    int(tokens[-1].findtext("CharacterOffsetEnd")),
                )
            )
        for line in files[name[:-4] + ".ann"].decode().splitlines():
            fields = line.split("\t")
            if len(fields) != 3 or not fields[1].startswith("Method "):
                continue
            audit["method_annotations"] += 1
            offsets = fields[1].split()[1:]
            if len(offsets) != 2 or not all(x.isdecimal() for x in offsets):
                audit["discontinuous_or_invalid"] += 1
                continue
            a, b = map(int, offsets)
            surface = fields[2]
            if doc[a:b] != surface:
                audit["text_mismatch"] += 1
                continue
            norm = " ".join(re.findall(r"[a-z0-9]+", surface.lower()))
            if not norm:
                continue
            postings[norm].add(key)
            for start, end in spans:
                if start <= a < b <= end:
                    mentions[key].append((a, norm, surface, doc[start:end]))
                    break
            else:
                audit["missing_sentence_boundary"] += 1
    visible, masked = [], []
    for key in sorted(docs):
        for _, norm, surface, text in sorted(mentions[key]):
            targets = {k: 1 for k in postings[norm] if k != key}
            if targets:
                pattern = r"(?<!\w)" + re.escape(surface) + r"(?!\w)"
                hidden = re.sub(pattern, "the technique", text, flags=re.I)
                if hidden == text:
                    raise ValueError(f"Annotated surface could not be masked in {key}")
                visible.append(Query(key, norm, text, {key}, targets))
                masked.append(Query(key, norm, hidden, {key}, targets))
                break
    audit.update(
        documents=len(docs),
        eligible=len(visible),
        selected_names=len({q.cluster for q in visible}),
    )
    return docs, visible, masked, dict(audit)


def acl_arc(cache):
    with archive(cache, "arc.tar.gz") as f:
        splits = {
            s: list(map(json.loads, f.extractfile(f"acl-arc/{s}.jsonl")))
            for s in ("train", "dev", "test")
        }
    aliases = nx.Graph()
    titles, raw_years = defaultdict(set), defaultdict(set)
    for rows in splits.values():
        for r in rows:
            title = r.get("cited_paper_title") or ""
            norm = title_key(title)
            pid = r["cited_paper_id"]
            if norm and pid:
                aliases.add_edge(("id", pid), ("title", norm))
                titles[norm].add(title)
                if r.get("cited_paper_year"):
                    raw_years[norm].add(int(r["cited_paper_year"]))
    docs, years, ids, canonical = {}, {}, {}, {}
    for component in nx.connected_components(aliases):
        names = [v for k, v in component if k == "title"]
        source_ids = {v for k, v in component if k == "id"}
        key = min(source_ids)
        ids[key] = source_ids
        docs[key] = min(
            (t for n in names for t in titles[n]), key=lambda t: (len(t), t)
        )
        years[key] = set().union(*(raw_years[n] for n in names))
        canonical.update({n: key for n in names})
    queries, audit = [], Counter()
    for r in splits["test"]:
        if r["intent"] != "Uses":
            continue
        audit["uses_test"] += 1
        target_title = title_key(r.get("cited_paper_title") or "")
        key = canonical.get(target_title)
        cleaned = r.get("cleaned_cite_text") or ""
        if "@@CITATION" not in cleaned:
            audit["missing_marker"] += 1
            continue
        text = cleaned.replace("@@CITATION", " ").strip()
        year = r.get("citing_paper_year")
        excluded = {
            k
            for k in docs
            if r["citing_paper_id"] in ids[k]
            or (year and years[k] and min(years[k]) > int(year))
        }
        if not text or not key or key in excluded:
            audit["ineligible"] += 1
            continue
        audit["target_title_inside_query"] += int(target_title in title_key(text))
        queries.append(
            Query(r["citation_id"], r["citing_paper_id"], text, excluded, {key: 1})
        )
    audit.update(
        eligible=len(queries),
        catalog_papers=len(docs),
        distinct_titles=len(titles),
        source_ids=len(
            {r["citing_paper_id"] for rows in splits.values() for r in rows}
        ),
        source_title_values=len(
            {r["citing_paper_title"] for rows in splits.values() for r in rows}
        ),
    )
    return docs, queries, dict(audit)
