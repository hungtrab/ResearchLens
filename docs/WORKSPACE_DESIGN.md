# Research workspace extension · 1.1

User brief: Projects connect a research topic to papers and citation graph; Paper Reading embeds the paper and shows related citations; Obsidian-style graph; light animation. Implemented on the existing FastAPI / vanilla HTML/CSS/JS stack, with no frontend build or paid service.

## Design reference and attribution

Applied the UI/UX guidance from [evondevKit](https://github.com/evondev/evondevKit), `skills/ui-ux/SKILL.md`, at commit `e2453581de25cf013f22dce0ff53a0e4c56ae3c0`. This is a design skill, not a runtime widget library. Read its design-process, principles, overlay, styles and scrollbar token references. MIT copyright notice is retained at `docs/third_party/evondevKit-LICENSE`; scrollbar styling derives from its tokens. No global agent configuration was changed.

User authorization to finish the functioning app took precedence over the skill's default wireframe approval gates and its visual-only scope. Existing ResearchLens colors, components and English app labels were retained. Project pages put the brief above the reading list; the reading view places the document beside notes. The layout targets desktop and stacks at narrower widths.

[Obsidian Graph documentation](https://obsidian.md/help/plugins/graph) informed the dark neutral surface, thin edges, neighborhood highlighting, groups/labels toggles, search, pan/zoom, draggable nodes and keyboard navigation. [Zotero Reader documentation](https://www.zotero.org/support/pdf_reader) informed the document-plus-notes arrangement. This is an independent implementation, not an Obsidian plugin or Zotero annotation engine.

## Shipped workflows

1. Create/edit/delete a Project; write a research brief up to 6,000 characters. An explicit example-brief button fills an editable example; it does not fabricate research findings.
2. Find related papers with BM25 from indexed English terms in the title/brief. Exclude project members and exact normalized-title duplicates from suggestions. This is query expansion from a brief, not an LLM assessment of relevance; no new effectiveness benchmark is claimed.
3. Attach/detach real corpus papers from suggestions, Discover or Reader. Progress derives from the shared reading status: To read, Reading, Read. Filter the project list by status.
4. Inspect an induced directed citation graph, optionally expanded with adjacent papers. Project membership colors nodes; only observed corpus citations create edges. Graph views are bounded to 200 nodes; project view defaults to100. Spatial layout is for readability, not a metric distance.
5. Open public PDFs directly in the browser's PDF viewer. Supported sources are exact HTTPS repository hosts listed in `researchlens/pdf.py`. Each redirect is revalidated, downloads are limited to20 MiB, streamed with timeouts and signature validation. HTML/challenges, unsupported hosts and unavailable PDFs fall back to a clearly labeled metadata abstract with an original-source link. No paywall bypass, arbitrary URL proxy or PDF text indexing.
6. Keep paper notes (30,000 characters), status and explicit page bookmark in SQLite. Notes save after650ms idle or Save. Large UTF-8 content uses ordinary fetch. Local drafts recover failed saves and route changes; draft keys are scoped by hash of profile+paper. Page bookmark is manual: scrolling native PDF does not update it. Browser PDF renderers vary, so original source remains available.
7. Focus mode expands the PDF. Export project brief/papers/notes or one paper's notes as JSON. Library BibTeX export remains available separately.
8. Graphs support title search, selection, spring-coupled node dragging, canvas pan, wheel/buttons/keyboard zoom, group colors and labels. Enter/Space selects a node, then Read paper opens its reader. Arrows retain citing→cited direction. Auto-label collisions are suppressed; showing every label can still be dense.
9. Short view/dialog/save transitions and a350ms opacity entrance. Dragging pins the selected node to the pointer; link springs transmit forces to neighbors and further connected nodes, while repulsion limits overlaps. A damped fixed-step simulation cools to a stop after release; no continuous background animation. `prefers-reduced-motion` disables these effects. Reduced motion preserves coupled direct manipulation through synchronous relaxation, with no release inertia. Simulation stops when its SVG is removed. Reset restores the initial layout.

Projects and reading records use the same anonymous cookie profile as the library. This is a local teaching deployment, not login/account access control. Deleting a project removes its memberships but preserves library and paper notes. Schema additions use `CREATE TABLE IF NOT EXISTS`, preserving existing volumes.

## Validation

- Independent API/store/PDF unit tests, including cross-profile isolation, persisted state, true citation edges, a full top100 suggestion window, PDF URL/redirect/byte/time/signature cases.
- `scripts/workspace_qa.py`: real corpus and actual public arXiv PDF, project creation/membership, note reload,22,000 Vietnamese characters, failed-save draft recovery, PDF fallback, graph interactions, mobile overflow and reduced motion.
- `scripts/browser_qa.py`: existing discovery/library/review/analytics flows.
- Independent Chromium audit: early-drag animation interruption, stale modal response, large UTF-8 note saving and draft recovery.

## Further ideas, not shipped

- Evidence matrix: compare task/dataset/method/limitations across selected papers, with manual verification and links to pages.
- Research-question checklist: attach a note or paper to each claim/gap in the brief; distinguish human assertions from source evidence.
- Temporal graph: restrict edges to a publication cutoff to avoid implying future knowledge.
- Structured BibTeX export for a single project, citation styles and duplicate-version reconciliation.
- PDF text annotations and full-text search require an actual extraction/annotation layer; current notes are separate from the PDF.

These ideas are deliberately separate from implemented features and measured mining results.

Graph-force regression: `scripts/graph_qa.py` measures direct/second-hop neighbor displacement, pointer pinning at non-unit zoom, cooling to rest, reset, reduced motion and detached-graph cleanup. See `artifacts/graph_qa.json`.
