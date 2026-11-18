# ResearchLens independent implementation audit

**Date:** 2026-10-14. **Status:** all seven original findings resolved and independently verified. No unresolved blocker was found for the supplied local classroom dataset. This is a scoped engineering review, not a production security certification.

The auditor was separate from the implementation and test authors and changed no implementation source. Scope: mining, API, SQLite storage, review modeling, frontend, collector, evaluation, and optional Solr integration.

## Original findings and verified fixes

| Priority | Original defect and reproduction | Verified resolution |
|---|---|---|
| P1 | API JSONL loading used `str.splitlines()`. Unicode separators inside review `PR743-r3` split a valid JSON string and blocked startup. | Loader splits physical LF only. Complete corpus starts successfully: 3,595 papers and trained review model. |
| P2 | Frontend silently sent only the first 50 saved papers as recommendation seeds, contradicting its exclusion claim. | Truncation removed; API accepts up to 10,000 seeds. A 52-paper fixture with 51 saved papers recommends only the remaining paper. |
| P2 | After corpus replacement, `/api/library` returned stale saved IDs even when its `items` list filtered them, causing recommendation HTTP 422. | Repeated save → restart with changed corpus → restore cookie. Returned IDs and items are empty; cold-start recommendation succeeds. |
| P2 | `--per-topic` consumed full 200-item pages, so declared provenance was wrong for nonmultiples of 200. | Temporary cached-data fixtures requesting 1 and 201 papers return exactly 1 and 201. Invalid limits are rejected. Each corpus file uses temporary-file replacement; no temporary files remain after completion. |
| P3 | Two differently labeled reviews with disjoint vocabularies crashed because `min_df=2` removed all features. | Same fixture now trains using `min_df=1` fallback. Punctuation-only reviews return an unavailable model gracefully. |
| P3 | Duplicate paper IDs silently desynchronized matrices, `pos`, and `by_id`. | A two-record duplicate-ID fixture now raises the explicit “Paper IDs must be unique” error before indexing. |
| P3 | An all-stopword corpus crashed with an internal sklearn empty-vocabulary error. | Empty and stopword-only corpora now fail with actionable “At least one paper” / “indexable words” validation messages. |

P1 blocks the supplied dataset; P2 changes supported behavior or experiment provenance; P3 concerns bounded input robustness.

## Data, API, and security evidence

Independent physical-line parsing found 3,595 unique paper IDs, 1,320 unique review IDs, and no orphan reviews. Review splits contain 1,082 train, 123 development, and 115 test reviews. No paper spans review splits, and no exact normalized review text crosses splits.

A fresh `TestClient` using the complete real corpus and a temporary SQLite database returned HTTP 200 for health, search, stats, facets, one-node graph, paper detail, paper reviews, review listing, experiment results, BibTeX export, saving, recommendations, and opinion analysis. A write with an unrelated Origin returned HTTP 403.

Reviewed controls include parameterized SQL; commit/rollback/close handling; HTML escaping; HTTP(S)-only external links; `rel="noopener"`; 256-bit random anonymous profile cookies with HttpOnly and SameSite Strict; Secure cookies under HTTPS; and a restrictive content-security policy. Local anonymous profiles are not authenticated accounts. Normal sparse recommendations, seed exclusion, empty-query browsing, OOV queries, simple AND, and quoted phrases worked in spot checks.

To reproduce the original loader issue independently:

```python
import json
from pathlib import Path
p = Path('data/reviews.jsonl')
# Incorrect: splits Unicode separators inside JSON strings.
# [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
with p.open(encoding='utf8') as f:
    rows = [json.loads(x) for x in f if x.strip()]
assert len(rows) == 1320
```

## Scientific and numerical audit

The audit includes `researchlens/evaluation.py`, `scripts/evaluate.py`, and the saved experiment outputs.

- All 120 saved reference tasks partition actual in-corpus references into disjoint seed/target sets. The target count is `floor(degree/3)`; 442 edges are hidden in total.
- The union of hidden edges is removed before rebuilding graph-derived features. Global and personalized PageRank use this masked graph. All methods exclude source and seed papers consistently.
- Review vocabulary and logistic-regression fitting use train reviews only. Official paper-disjoint splits are asserted. Review confidence intervals resample test-paper clusters.
- Binary nDCG, AP denominators, P@10, Recall@10, first-hit MRR, and paired recommendation bootstrap were inspected. A hand-check ranking `a,b,c` with relevant `a,c` produces AP 5/6, nDCG@10 0.91972, and MRR 1.
- Corpus and review hashes match the saved run. All retrieval and recommendation aggregate means were independently recomputed from saved observation rows and matched.

| Comparison | Saved evidence | Supported interpretation |
|---|---|---|
| BM25 vs TF-IDF, SciFact | nDCG@10 0.6514 vs 0.6209; Recall@10 0.7678 vs 0.7777 | BM25 improves ranking quality; TF-IDF has slightly higher recall at ten. |
| Graph vs hybrid recommendation | nDCG@10 0.1744 vs 0.1119 | Graph-only is strongest on this citation-recovery task. |
| Hybrid vs content recommendation | Paired nDCG difference +0.0489; 95% CI [0.0274, 0.0734] | The experiment supports improvement over content alone. |
| Logreg vs aspect rules | Macro-F1 0.4690 vs 0.4362; broad overlapping intervals | Report descriptive results; no demonstrated significant advantage over rules. |

Citation direction is citing → cited. External citation counts remain distinct from local graph metrics. Collection-query labels are acquisition labels, not relevance judgments. Citation recovery is a transductive, non-temporal proxy rather than user satisfaction. Review labels derive from ratings, not independently annotated sentiment or scientific quality. Aspect and entity extraction lack a gold annotated benchmark; no validated accuracy claim is supported for those components.

Retrieval omits zero-score documents, so AP refers to an untruncated retrieved list rather than an arbitrary complete-corpus tie ordering. The fixed reported run uses 120 recommendation tasks; any alternate task count must be reflected accurately in its protocol text.

## Optional Apache Solr audit

Reviewed `researchlens/solr.py`, `scripts/solr_index.py`, and API routing. Independent mocked checks verified result mapping, year/topic/venue/OA filters, timeout configuration, and rejection of returned unknown paper IDs. Input is quoted and escapes Lucene special syntax; a probe such as `{!lucene}*:*` becomes literal quoted text rather than parser parameters. The Solr host is operator-configured, not a request-controlled URL. API access is read-only; indexing is a separate script.

The indexer defines explicit fields and indexes metadata plus local PageRank. It intentionally replaces all documents in the configured core: it must point only to the dedicated ResearchLens core. Live Docker/Solr execution is being checked by the implementing agent and is not represented as an independently executed audit here.

Remaining operational limits are explicit:

- Solr `text_en` analysis and field boosting differ from the local tokenizer/title repetition; rankings and scores need not match. The saved SciFact benchmark evaluates the local implementations, not Solr.
- Unknown-ID detection catches some stale indexes but cannot detect missing documents or changed metadata under the same IDs. Reindex after changing the corpus.
- Corpus output replacement is atomic per file, not a transaction across papers, reviews, and provenance. Finish collection before starting the application.
- The 10,000-seed request bound comfortably covers the supplied 3,595-paper snapshot; substantially larger deployments need a server-side profile recommendation endpoint or an explicit larger limit.

## Version 1.1 follow-up: projects, reader, and citation workspace

Independent review scope: `workspace.py`, `pdf.py`, new persistence methods, API integration, and the new frontend modules. Findings below concern the expanded feature set and do not reopen the seven resolved baseline findings.

### Findings and verification status

| Priority | Finding | Status |
|---|---|---|
| P2 | Project suggestions ranked only 100 results before excluding attached papers. A fixture with 101 matching papers and its first 100 attached returned no suggestion despite one eligible paper. | Resolved: a 101-paper fixture with distinct matching titles and 100 attached papers suggests the remaining paper; repeated-title versions are intentionally deduplicated. |
| Integration check | Direct PDF iframe embedding would conflict with `frame-ancestors 'none'`. | Reader fetches the PDF and embeds a local blob URL, avoiding direct response embedding; fallback was verified in Chromium. |
| P2 | Every note save used `keepalive:true`, rejecting valid Unicode note bodies above the browser 64 KiB limit. A 22,000-character Vietnamese fixture failed before network delivery. | Resolved: ordinary fetch successfully saved a 66,039-byte payload in Chromium. |
| P2 | Failed saves followed by navigation discarded the detached reader form and its unsaved notes. | Resolved: synchronous local drafts survive navigation; reopening recovers text; successful matching save clears the draft. Keys are profile- and paper-specific. |
| P2 | Dragging the graph during its first 650 ms left an inline CSS scale overriding later SVG zoom/pan. | Resolved: Chromium early-drag reproduction now leaves no overriding CSS transform. |
| P3 | A pending picker response could replace a newer project form opened in the same dialog. | Resolved: initial GET, successful mutations, and failed mutation callbacks now check modal generation. Chromium verified that an old failed request preserves a newer loading modal without JavaScript errors. |

### Backend checks completed

- Two independent browser profiles were exercised against project detail, modification, deletion, paper attachment, suggestions, and graph routes. Attempts by the other profile returned 404; reading notes/list remained private to the original profile.
- Project memberships and reading notes survived app restart with the same SQLite database and profile cookie.
- A bounded project graph included only selected project/neighbor nodes, and every emitted edge connected emitted nodes. Citation edges come directly from the observed directed graph; proximity in the layout does not create inferred citations.
- PDF URL tests reject HTTP, loopback IPs, misleading hostname suffixes, alternate ports, embedded credentials, fragments, and backslash authority tricks.
- A mocked allowed-host redirect to `http://127.0.0.1:9000/private` was rejected before a second request. Redirect targets are revalidated, automatic redirect following is disabled, and environment proxy settings are ignored.
- Mocked HTML responses and oversized Content-Length were rejected; valid `%PDF-` content was accepted. Download limits also inspect accumulated decoded bytes, rather than trusting Content-Length alone.

The PDF proxy accepts only metadata-selected HTTPS URLs on its fixed public repository allowlist; no user-facing arbitrary URL fetch endpoint is exposed. It does not bypass publisher access restrictions. The transport timeout and chunk deadline bound ordinary slow responses, but the nominal 25-second deadline is checked between yielded content chunks rather than being a strict cancellation timer.

### Frontend verification evidence

A Chromium session against the 1.1 UI used mocked reading writes to avoid changing the shared database. It successfully saved 22,000 Vietnamese characters, recovered a failed-save draft after navigating away and back, and removed the local draft after a successful retry. No JavaScript errors occurred. Separate isolated browser fixtures verified stale picker GET protection and graph transform cleanup after a drag at 80 ms. Backend checks verified stable draft identifiers that differ across both profiles and papers.

The reader displays an explicit abstract fallback when PDF retrieval fails, distinguishes that abstract from the full paper, and exposes the original source. Page bookmarks are manual and do not claim to track scrolling in the browser PDF viewer.

A further real-PDF browser fixture served a local PDF with `frame-ancestors none`; the reader created its blob iframe without a CSP ancestor error and removed it on navigation. Delayed successful picker mutations were also verified not to close a newer project form.

### Final 1.1 disposition

All reported 1.1 findings are resolved. The final callback error guards were reread, and an isolated Chromium failure reproduction passed without replacing the new dialog or raising JavaScript errors. The final distinct-title suggestion fixture also passed after title deduplication. Deduplication affects suggestion presentation only; paper IDs and citation edges remain intact. PDF bookmark navigation now replaces the iframe to reload the requested page; page-4 rendering was verified by the implementing agent.

## Version 1.2 follow-up: conference catalog and online discovery

Scope: `catalog.py`, `sources.py`, `discovery.py`, import/alias persistence, startup integration, `scripts/crawl_conferences.py`, and `static/discovery.js`. All reproductions use temporary databases and mocked remote metadata. The active crawl and production data were not modified or stopped.

### Findings

| Priority | Finding and reproduction | Status |
|---|---|---|
| P1 | Legacy official virtual-program rows lack `paper_url` and `virtualsite_url`; fallback hashes the same edition index URL for every paper. Cached-only parsing shows NeurIPS 2020: 2,004 → 1 record, 2021: 2,649 → 1; ICLR 2020: 687 → 1, 2021: 913 → 1, 2022: 1,150 → 1 after merge. | Parser resolved and verified against cached official data: NeurIPS 2020/2021 retain 1,899/2,334 distinct papers; ICLR 2020/2021/2022 retain 687/860/1,095. Final merged snapshot verified: every parsed legacy ID survives as a canonical or alternate ID. |
| P2 | NeurIPS 2022 official data includes 48 parent session events labeled `Spotlight`, such as “Lightning Talks 4B-1”, with nonempty `children_ids` and `/session/` URLs. The original filter treated them as papers. | Resolved: all 48 parent/session events are excluded by cached-data verification. |
| P2 | Graph construction ignores persisted external-ID aliases. Import official record `OFFICIAL-X`, deduplicate matching `W1` to it, then import `W2` referencing `W1`: the graph lacks the observed `W2 → OFFICIAL-X` edge. | Resolved: the observed edge is present for both import orders and after restart; source payloads and benchmark files remain unchanged. |
| P2 | Remote metadata is insufficiently normalized before persistence. A string `publication_year: "2025"` imports successfully, then `/api/stats` and `/api/facets` return 500 because years mix strings and integers. Null nested author objects and abstract-position arrays also escape as raw exceptions. | Resolved: year strings normalize to integers; malformed nested structures produce controlled upstream errors; stats and facets remain HTTP 200 after import. |
| P3 | Successful import sets `state.stats=null` while an already-started graph/analytics render may resume and dereference it. | Resolved: imports refresh statistics without assigning null. An independent Chromium import fixture verified object refresh and reader navigation. |
| Clarity | Imports change the shared mining corpus, whereas notes/projects/library are private. Current workspace statistics differ from the fixed benchmark after imports. | Resolved: shared-corpus/private-profile boundaries are explicit, and analytics distinguishes live statistics from fixed benchmark measurements. |

### Verified properties

- Five simultaneous distinct imports preserve all five entries; five simultaneous imports of one ID create exactly one new workspace entry. This verifies the single-process import lock and immutable-engine replacement path.
- Imported papers survive restart. Temporary benchmark JSONL bytes remain unchanged after import, and the evaluation script still reads the original snapshot files directly.
- Catalog SQL values are parameterized. FTS input is tokenized and quoted; adversarial quotes, `OR *`, `NEAR`, field syntax, SQL-like strings, Unicode, and punctuation return controlled responses without SQL/parser exceptions.
- Remote discovery uses a fixed OpenAlex origin and ID validation. Import requests contain only source/ID/enrichment choice; the browser cannot submit arbitrary paper payloads or fetch URLs.
- Frontend metadata is escaped; source links use the existing HTTP(S) URL filter. Cached upstream objects are deep-copied before response decoration.
- The catalog is a disk-backed FTS index distinct from the runtime mining engine. Its sync replaces rows and rebuilds the FTS index within a transaction.
- Coverage text states that counts are collected records, not independently verified totals of accepted papers. The crawl records unavailable/unpublished sources and its source URLs; only completed source jobs enter the final merged corpus.
- Cached official PMLR page headers independently confirm that volumes 119, 139, 162, 202, 235, 267, and 306 correspond to ICML 2020 through 2026. Final eight-conference coverage totals were independently verified against the completed snapshot.

The import lock coordinates threads in one app process. Multiple independent worker processes would each retain their own in-memory engine; the supported classroom deployment should continue to use one worker unless cross-process refresh is implemented.

The cached CVPR 2020 `?day=all` response contains an explicit SQL date error for `all`, although its HTTP status is 200. This explains the recorded unavailable-source status; a bare-year index fallback was recommended. Legacy virtual-program `paper_pdf_url` fields sometimes contain OpenReview forum or NeurIPS abstract HTML URLs, so their names alone do not prove PDF content.

### 1.2 remediation checks

Independent Chromium fixtures render malicious remote title/author HTML as text, disable a `javascript:` source link, submit only source and ID when importing, refresh statistics, and navigate to the canonical workspace paper. No JavaScript errors occurred. Import/alias regression checks pass in both orders, including restart. Parser corrections retain legacy paper identities and exclude program sessions. CVPR 2020 now enumerates the three linked day indexes; cached extraction independently returns 1,466 unique records from the landing page plus three day sources.


### Final 1.2 disposition

All reported 1.2 findings are resolved. Independent read-only validation of the completed snapshot found:

- **129,596 catalog records**, matching the manifest count and SHA-256 exactly.
- **Zero duplicate IDs** and **zero duplicate normalized titles within a conference/year**.
- All **56 conference/year coverage cells** match the JSONL records; basic metadata types and the 2020–2026 year bounds pass.
- Every parsed legacy NeurIPS 2020–2022 and ICLR 2020–2022 ID is preserved as either a canonical catalog ID or an alternate ID.
- The corrected CVPR 2020 parser returns **1,466 records** after following the three dated indexes and deduplicating paper IDs.
- No source job is marked unavailable. Empty recorded editions are **EMNLP 2026** and **NAACL 2020, 2023, and 2026**; their absence is explicit and must not be described as complete coverage of all accepted papers.
- Original benchmark paper and review file hashes still match `artifacts/results.json`; the expanded discovery catalog has not changed the measured experiment inputs.

The remaining limits are documented design boundaries: collected-source coverage is not an independently certified acceptance census, imports are shared across the local runtime, private reading/project state remains profile-scoped, and runtime engine refresh is coordinated within a single app process.
