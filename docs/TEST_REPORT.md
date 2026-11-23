# ResearchLens automated test report

Executed on 18 November 2026 with Python 3.11.16 in the project's `uv` environment.

```sh
uv run pytest -q --cov=researchlens --cov-report=term-missing --junitxml=artifacts/pytest.xml
```

**Result: 574 passed, 0 failed in 87.13 seconds.** One dependency deprecation warning was emitted by Starlette's TestClient about its HTTPX integration; it did not affect the tests.

Current machine-readable results are in `artifacts/pytest.xml`; the console and coverage record is `artifacts/test-results.txt`.

The suite uses six base paper records with explicit citation edges, five review records, purpose-built catalog/source fixtures, fresh temporary SQLite files, and FastAPI's in-process TestClient with application startup/shutdown. It makes no network requests and does not modify the collected corpus or the running application's database. These fixtures test correctness; experimental performance and retrieval quality numbers come from the separate evaluation pipeline on the collected dataset.

| Module | Statement coverage |
|---|---:|
| `researchlens/api.py` | 99% |
| `researchlens/mining.py` | 99% |
| `researchlens/reviews.py` | 91% |
| `researchlens/store.py` | 99% |
| `researchlens/evaluation.py` | 100% |
| `researchlens/solr.py` | 100% |
| `researchlens/workspace.py` | 100% |
| `researchlens/pdf.py` | 100% |
| `researchlens/catalog.py` | 100% |
| `researchlens/sources.py` | 100% |
| `researchlens/discovery.py` | 100% |
| `researchlens/passages.py` | 99% |
| Total | **99%** (1,168 / 1,180 statements) |

`tests/test_fusion.py` adds 10 behavior cases for the default fusion recommender: default method in the engine and the API, a candidate strong in both rankings beats one strong in only one, seeds never appear, co-cited items carry witnesses, non-co-cited items still fill the list, and diversity is respected. Expected values are worked out by hand from a small fixture. Seven deliberate code mutations each made at least one fusion test fail.

The method retrieval study adds 33 independent offline regression cases covering BM25, phrase order, reciprocal-rank ties, shortlist stability, graded metrics, source identity, date filtering, masking and Laya scoring. A regression exposed mismatched candidate-vector lengths silently broadcasting during fusion. The evaluator now rejects that input.

The November audit added seven independent co-citation tests and one regression for AND across whitespace. A Chromium check with in-process API routing verified that clicking an evidence paper in My library opens its matching detail view.

## Verified behavior

- Retrieval: exact topic outranks partial matches for BM25 and TF–IDF; BM25 scores match a separately calculated formula; phrase constraints, explicit AND, default OR, exclusions, empty queries, unknown terms, filtering and pagination work. Exclusions respect word boundaries (`-graph` retains `paragraph`); quoted phrases respect word boundaries and allow whitespace differences. Citation ranking still filters for query relevance. A corpus without indexable words fails with an actionable error.
- Citation mining: edges point from citing to cited papers; out-of-corpus references and self-citations do not create edges; the cited authority receives higher PageRank/HITS authority; hub scores, normalization, isolates and focus subgraphs are checked.
- Recommendation: the four baseline methods exclude seed papers, remove duplicate seeds, remain deterministic and expose finite explanations. Cold start, content similarity, citation proximity, full-corpus seed sets, maximal-diversity redundancy reduction, unknown seeds, and a library with 51 distinct seeds are exercised.
- Reviews and extraction: sentence evidence remains verbatim, negation reverses polarity, aspects are attached to evidence, and every extracted entity's offsets recover its exact text, including after Unicode characters. Deduplication is case/whitespace invariant within a paper and retains reviews attached to different papers.
- Classifier: only training-split reviews enter the vocabulary, predictions contain normalized probabilities, absent/single-class training sets are handled, and a tiny two-class corpus without shared vocabulary does not prevent operation.
- Persistence: SQLite survives reopening, saves are idempotent, profiles and histories are isolated, history is bounded and ordered, foreign keys reject unknown papers, rollback works, metadata updates preserve the library, and SQL-like query text is stored literally.
- API: lifecycle, health, static entry page, search/detail/graph consistency, facets/statistics, reviews, opinion and extraction, input validation, missing resources, session cookies, library persistence, JSON/BibTeX export and escaping, cross-origin write rejection, and same-origin writes are checked. Stale saved-paper IDs are omitted from libraries and exports after corpus replacement. Optional review data and raw Unicode U+2028/U+2029 inside JSONL fields have regression coverage.
- Optional Solr: mocked HTTP checks verify enabled/disabled modes, connection and HTTP failures, stale index IDs, Lucene escaping, local-parameter injection defense, phrase/AND/exclusion translation, field/boolean/year filters, open-ended years, paging, scores and empty-query PageRank ordering. Tests clear inherited `SOLR_URL` configuration so a local service cannot affect their outcome. A regression exposed malformed upstream JSON being misclassified as a 422 user error; the fixed API now reports 503 while keeping malformed user queries at 422.
- Evaluation: known ranks independently verify precision, recall, nDCG, AP and reciprocal rank. Relevant results beyond rank 10 contribute to full-ranking AP. Empty/duplicate rankings and absent relevance are checked. Bootstrap intervals are reproducible, and aggregation keeps methods separate.

## Projects and PDF reader expansion

The added `tests/test_workspace.py` and `tests/test_pdf.py` contribute **73 passing cases**, included in the 270-test total. No application code was changed by the testing agent, and no production database or publisher service was accessed.

- Projects: create, list, retrieve, update and delete; title/brief validation; idempotent membership; member removal; project deletion cascades membership while retaining independent reader notes. Both API and direct Store tests prevent another profile from reading, updating, deleting or adding papers to a project.
- Suggestions: brief terms produce relevant BM25 suggestions, existing members are excluded, unknown vocabulary produces an empty result, and natural prose containing unclosed quotation marks or Boolean-looking text does not become a malformed search expression. A 111-paper regression checks that suggestions still return ten eligible matches when the first 100 ranked papers are already project members. Title normalization excludes case/punctuation variants of project members and collapses duplicate suggestion titles while retaining distinct titles; the citation graph retains every original paper ID and edge.
- Project citation graph: every returned edge is a real citing-to-cited corpus edge; membership flags distinguish members from neighbors; disconnected members remain visible; optional neighbors, empty projects, node limits and finite coordinates are checked.
- Reading workspace: queued defaults, status/page/note validation, Unicode and literal HTML-like note text, persistence across application restart, profile isolation, project hydration with reading state, stale corpus filtering, missing-paper handling and cross-origin write rejection. Draft recovery keys remain stable across reads, updates and application restarts, differ between papers and profiles, and do not expose the raw session cookie.
- PDF reader: HTTPS repository allowlist, user-info/port/fragment/control-character/hostname-spoofing rejection, and pre-network rejection of unsupported URLs. Mocks verify disabled automatic redirects and environment proxies, relative and supported cross-repository redirects, rejection before contacting an unsupported redirect target, bounded redirect loops, declared and streamed byte limits, deadline checks, PDF signature validation, upstream HTTP/connection/timeout errors, and inline API response behavior. These tests validate transport policy and response handling; they do not claim that every allowlisted repository is always reachable or that its PDF viewer renders correctly.

## Conference catalog, online discovery and import expansion

`tests/test_discovery.py` adds 70 cases. All OpenAlex and official-publisher interactions in these tests are mocked; neither the active conference crawl nor production datasets are modified.

- Catalog: FTS5 relevance and filters, year/conference/offset validation, quoted or SQL-like text, snapshot replacement, persistence, missing snapshots, and transaction rollback after a duplicate appears beyond the first 1,000-row insert batch.
- Source normalization: authors, inverted abstracts, citations, references, venue/conference identity and preferred OA links; missing optional fields and invalid IDs; strict malformed-payload handling. Regression tests verify controlled 503 responses without workspace mutation and ensure a numeric year string cannot break local filters, facets or statistics.
- Online service: query/filter/paging parameters, HTTP/quota/connection/JSON failures, bounded five-minute cache expiry, deep-copy isolation, and API keys carried in headers rather than query strings. Viewing live results does not import them or alter the fixed benchmark count.
- Explicit import: restart persistence, DOI/title deduplication, external aliases, official metadata retention during enrichment, fallback warnings, and citation edges to canonical imported IDs. Alias-edge regressions run in both import orders and again after restart. Concurrent imports of the same paper commit once; distinct concurrent imports both survive. Simulated persistence failure leaves the live engine unchanged. Shared imported metadata does not expose another profile's library, projects or reading notes.
- Official adapters: virtual-program event filtering and PDF derivation, distinct old-schema event identities, exclusion of parent Spotlight sessions, PMLR volume parsing, ACL mixed-markup titles and frontmatter exclusion, conference-specific volumes in the shared Findings XML, CVPR relative links, the 2020 bare-index fallback, and 2020 day-navigation enumeration with repeated-link/paper deduplication, cursor exhaustion/repetition protection, and deduplication that retains OpenAlex citation identities while merging official metadata.
- Reader extension: eight extra safety cases cover CVF and ICLR PDF hosts and the restricted raw GitHub `mlresearch` PDF path; other organizations, non-PDF files and literal or encoded parent-directory escapes are rejected.

The added tests reproduced and verified fixes for malformed upstream nested objects, year typing, missing citation edges through imported aliases, collapsed historical virtual-program IDs, parent session entries and historical proceedings index formats. They do not establish that a conference crawl is complete; actual crawl coverage and source failures remain recorded in its manifest.

## Scope and remaining limitations

Statement coverage is not a proof of ranking quality or complete branch coverage. Live-service availability, collector transport retry timing, sustained load, browser rendering, and actual scholarly relevance are outside this isolated suite. The experiments endpoint's artifact-present/absent paths and rare classifier fallback error paths are not included in the reported coverage. Browser interaction and screenshot verification are performed separately by the integration agent.

The suite intentionally does not treat lexicon-based sentiment as authoritative review interpretation. It verifies the documented heuristic and evidence spans. Mocked Solr checks establish the adapter contract. The integration agent separately ran Docker Solr 9.9.0 against 3,595 indexed papers; `artifacts/solr_integration.json` records six checks covering empty queries, ordinary terms, quoted phrases, AND with filters, negative-only queries and term exclusion. That integration run is separate from the 270 isolated tests and is not included in their timing or coverage.

## Earlier packaging verification (baseline release)

The primary agent repeated the unchanged 127-test suite on the final source after adding the offline API reference: **127 passed in 3.06 seconds, 99% coverage (421/427 statements)**. The test artifacts have since been refreshed by the current run above. The two new documentation-route statements were separately checked in a real browser against `/docs`. The final browser run passed all recorded flows with no JavaScript errors or mobile horizontal overflow; see `artifacts/browser_qa.json`. `scripts/verify_artifacts.py` additionally verifies corpus hashes, IDs, splits, masked targets and saved aggregate metrics.


## Earlier handoff verification (Projects release)

Primary integration run: **192 passed**, 606/612 statements covered (99%), 8.32 seconds. The JUnit artifact now reflects the current run above. Both browser suites passed on the rebuilt Docker app at port8765 with Solr enabled. The workspace suite used the real public arXiv PDF for W3007332492; it covered project creation, membership, citation map, PDF rendering, manual bookmark, large UTF-8 notes, reload, failed-save recovery, missing-PDF fallback, graph controls, mobile layout and reduced motion. Results are in `artifacts/workspace_qa.json`; screenshots in `artifacts/screenshots/`.


## Coupled graph dragging

`scripts/graph_qa.py` adds a browser regression for spring-coupled dragging. It verifies direct and second-hop neighbor displacement, pointer pinning at non-unit zoom, damping to a complete stop after release, reset to the initial layout, reduced motion without release inertia, and cleanup after SVG removal. Browser results are recorded in `artifacts/graph_qa.json`; these checks are separate from the Python unit suite.

## Version 1.2 live integration

The final Docker deployment was checked at port 8765: conference BM25/filter/coverage and live OpenAlex search render without JavaScript errors. The full discovery acceptance flow ran against a disposable database on port 8767: online search, import, reading, notes reload, project attachment, citation-map membership and mobile layout all passed. `artifacts/discovery_qa.json` and `artifacts/discovery_deployment.json` record these results separately from the 270 isolated tests.

A real ICML 2025 catalog paper (`PMLR-v267-su25d`) was enriched to OpenAlex `W4415238825`, imported and read through the restricted mlresearch PDF host path. The returned 571,515-byte document passed the PDF signature check; evidence is in `artifacts/catalog_import_live.json`. Graph coupling and baseline browser suites also passed on the final Docker app.
