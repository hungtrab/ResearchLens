"""Report figures, drawn only from saved artifacts, in the figures4papers house style.

Style (github.com/ChenLiu-1996/figures4papers, scientific-figure-making skill): an Arial-like sans font, no top or
right spines, frameless legends, black bar edges, values printed on the bars, the blue/green/red/neutral palette
(blue for the method the figure is about, red for baselines, neutral for reference), tight layout, vector PDF plus
a 300 dpi PNG. Figures carry no titles; the LaTeX captions say what they show.
Run: uv run python scripts/report_figures.py  (writes artifacts/figures/report/)"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from researchlens.mining import MiningEngine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/figures/report"
PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_1": "#DDF3DE", "green_2": "#AADCA9",
           "green_3": "#8BCF8B", "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
           "neutral": "#CFCECE", "gray": "#767676", "dark": "#272727", "teal": "#42949E", "violet": "#9A4D8E"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#FFFFFF", PALETTE["blue_secondary"], PALETTE["blue_main"]])
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 15,
                     "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 2,
                     "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
SUITES = ("E2", "E2b", "E3")


def load(name):
    return json.loads((ROOT / "artifacts" / name).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def bars(ax, labels, values, colors, fmt="{:.3f}", errors=None, fontsize=12):
    x = np.arange(len(values))
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=1.5, width=0.7)
    if errors is not None:
        low, high = np.array(errors).T
        ax.errorbar(x, values, yerr=[np.array(values) - low, high - np.array(values)], fmt="none", ecolor="black",
                    capsize=4, lw=1.5)
    tops = np.array(errors)[:, 1] if errors is not None else np.array(values)
    for i, v in enumerate(values):
        ax.text(i, tops[i] + 0.012 * max(tops), fmt.format(v), ha="center", va="bottom", fontsize=fontsize)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(tops) * 1.18)


def grouped(ax, groups, series, labels, colors, ylabel, fontsize=11, hatches=None):
    """Bars for each group (suite), one bar per series, values printed on top."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    top = max(max(s) for s in series)
    for i, (values, label, color) in enumerate(zip(series, labels, colors)):
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, values, width, color=color, edgecolor="black", linewidth=1.2, label=label,
               hatch=hatches[i] if hatches else None)
        for p, v in zip(pos, values):
            ax.text(p, v + 0.01 * top, f"{v:.3f}".lstrip("0"), ha="center", va="bottom", fontsize=fontsize, rotation=90)
    ax.set_xticks(x, groups)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, top * 1.28)


def signals():
    """E2 and E2b nDCG@10 for every recommender of the app plus the logistic regression reranker."""
    temporal, author, ltr = load("temporal_recovery.json"), load("author_next.json"), load("ltr.json")
    names = [("popular", "Global\nPageRank"), ("content", "Content"), ("graph", "PPR"), ("hybrid", "Hybrid"),
             ("cocitation", "Cocitation"), ("fusion", "Fusion")]
    colors = [PALETTE["neutral"], PALETTE["red_1"], PALETTE["red_1"], PALETTE["red_2"], PALETTE["green_2"],
              PALETTE["blue_main"], PALETTE["green_3"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, data, run, title in ((axes[0], temporal, ltr["reference_recovery"], "E2, hidden references"),
                                 (axes[1], author, ltr["author_next"], "E2b, what an author cites next")):
        values = [data["summary"][k]["ndcg"] for k, _ in names] + [run["ndcg"]]
        bars(ax, [n for _, n in names] + ["Logistic\nregression"], values, colors, fontsize=12)
        ax.set_ylabel("nDCG@10")
        ax.set_xlabel(title, labelpad=10)
        ax.tick_params(axis="x", labelsize=12)
    save(fig, "signals")


def decision():
    """Fusion, the decision model and the decision model with a topic: nDCG@10 and Hit@10 on the three suites."""
    model, topic = load("decision_model.json")["test"], load("decision_query_model.json")["test"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = [PALETTE["red_2"], PALETTE["blue_secondary"], PALETTE["blue_main"]]
    labels = ["Fusion", "Decision model", "Decision model + topic"]
    grouped(axes[0], SUITES, [[model[s]["fusion_ndcg"] for s in SUITES], [model[s]["ndcg"] for s in SUITES],
                              [topic[s]["ndcg"] for s in SUITES]], labels, colors, "nDCG@10")
    temporal, author, holdout = load("temporal_recovery.json"), load("author_next.json"), load("holdout_fusion.json")
    hit = lambda tasks: np.mean([bool(set(t["methods"]["fusion"]["top10"]) & set(t["targets"])) for t in tasks])
    fusion_hit = [hit(temporal["observations"]), hit(author["observations"]), holdout["hits"]["fusion"]["hit10"]]
    grouped(axes[1], SUITES, [fusion_hit, [model[s]["hit"] for s in SUITES], [topic[s]["hit"] for s in SUITES]],
            labels, colors, "Hit@10")
    axes[1].legend(loc="upper left", fontsize=12, ncol=1)
    save(fig, "decision")


def rankers():
    """Language models as rankers on the same 20 and 100 fusion candidates, against the decision model."""
    slm20, text20, graph20 = load("slm_ranker.json"), load("llm_ranker.json"), load("llm_ranker_graph.json")
    slm100, graph100 = load("slm_ranker_pool100.json"), load("llm_ranker_graph_pool100.json")
    small = lambda result: next(m for n, m in result["models"].items() if "-1.5b-" in n)["suites"]
    ndcg = lambda per_suite: [per_suite[s]["ndcg"] for s in SUITES]
    base = lambda result, key: [result["baselines"][s][key]["ndcg"] for s in SUITES]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    grouped(axes[0], SUITES, [base(text20, "fusion"), base(text20, "decision_pool"), ndcg(small(slm20)),
                              ndcg(text20["suites"]), ndcg(graph20["suites"])],
            ["Fusion", "Decision model", "SLM 1.5B", "LLM, text only", "LLM + graph RAG"],
            [PALETTE["neutral"], PALETTE["blue_main"], PALETTE["red_1"], PALETTE["red_2"], PALETTE["green_3"]],
            "nDCG@10", fontsize=10)
    axes[0].set_xlabel("20 candidates", labelpad=8)
    grouped(axes[1], SUITES, [base(graph100, "fusion"), base(graph100, "decision_pool"), ndcg(small(slm100)),
                              ndcg(graph100["suites"])],
            ["Fusion", "Decision model", "SLM 1.5B", "LLM + graph RAG"],
            [PALETTE["neutral"], PALETTE["blue_main"], PALETTE["red_1"], PALETTE["green_3"]], "nDCG@10", fontsize=10)
    axes[1].set_xlabel("100 candidates", labelpad=8)
    axes[1].set_ylabel("")
    top = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, top)
    axes[0].legend(loc="upper left", fontsize=11, ncol=2)
    save(fig, "rankers")


def cost():
    """Mean nDCG@10 over the three suites against the median time per request (log scale)."""
    latency = load("latency_recommend.json")["summary"]
    slm20, text20, graph20 = load("slm_ranker.json"), load("llm_ranker.json"), load("llm_ranker_graph.json")
    slm100, graph100 = load("slm_ranker_pool100.json"), load("llm_ranker_graph_pool100.json")
    mean = lambda per_suite, key=None: np.mean([(per_suite[s][key] if key else per_suite[s])["ndcg"] for s in SUITES])
    model = lambda result, size: next(m for n, m in result["models"].items() if f"-{size}-" in n)
    points = [("Fusion", latency["fusion"]["p50_ms"] / 1000, mean(text20["baselines"], "fusion"), PALETTE["gray"]),
              ("Decision model (20)", latency["decision"]["p50_ms"] / 1000, mean(text20["baselines"], "decision_pool"),
               PALETTE["blue_main"]),
              ("Decision model (100)", latency["decision"]["p50_ms"] / 1000,
               mean(graph100["baselines"], "decision_pool"), PALETTE["blue_main"])]
    for size, label in (("0.5b", "SLM 0.5B"), ("1.5b", "SLM 1.5B")):
        for result, pool in ((slm20, 20), (slm100, 100)):
            m = model(result, size)
            points.append((f"{label} ({pool})", m["seconds_per_request"]["p50"], mean(m["suites"]), PALETTE["red_strong"]))
    for result, label in ((text20, "LLM, text (20)"), (graph20, "LLM + graph RAG (20)"),
                          (graph100, "LLM + graph RAG (100)")):
        points.append((label, result["seconds_per_request"]["p50"], mean(result["suites"]), PALETTE["teal"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = {"Decision model (100)": (8, 6), "Decision model (20)": (8, -16), "Fusion": (8, -4),
               "LLM + graph RAG (100)": (-165, 6), "LLM + graph RAG (20)": (12, -5), "LLM, text (20)": (-30, -24)}
    for label, seconds, value, color in points:
        ax.scatter(seconds, value, s=140, color=color, edgecolor="black", linewidth=1.2, zorder=3)
        ax.annotate(label, (seconds, value), xytext=offsets.get(label, (8, 4)), textcoords="offset points", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("Median seconds per request (log scale)")
    ax.set_ylabel("Mean nDCG@10 over E2, E2b, E3")
    ax.set_xlim(0.01, 5000)
    ax.set_ylim(0, 0.28)
    save(fig, "cost")


def retrieval_and_reviews(results):
    rows = results["retrieval"]["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
    bars(axes[0], ["TF-IDF", "BM25"], [r["ndcg_at_10"] for r in rows], [PALETTE["red_2"], PALETTE["blue_main"]],
         errors=[r["ndcg_at_10_ci95"] for r in rows])
    axes[0].set_ylabel("nDCG@10")
    axes[0].set_xlabel("SciFact, 300 test queries", labelpad=8)
    rows = results["opinion"]["summary"]
    bars(axes[1], ["Majority class", "Aspect lexicon", "TF-IDF +\nlogistic regression"], [r["macro_f1"] for r in rows],
         [PALETTE["neutral"], PALETTE["red_2"], PALETTE["blue_main"]], errors=[r["macro_f1_ci95"] for r in rows])
    axes[1].set_ylabel("Macro F1")
    axes[1].set_xlabel("PeerRead ICLR 2017, 115 test reviews", labelpad=8)
    save(fig, "retrieval_reviews")
    fig, ax = plt.subplots(figsize=(6.4, 5))
    matrix = np.array(rows[-1]["confusion_matrix"])
    image = ax.imshow(matrix, cmap=BLUES, vmin=0)
    labels = results["opinion"]["labels"]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Label from the review score")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=18,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, label="Test reviews")
    save(fig, "confusion")


def holdout(results):
    """E3 random holdout: nDCG@10 with intervals, and the diversity trade-off of MMR."""
    rows = results["recommendation"]["summary"]
    names = {"popular": "Global\nPageRank", "content": "Content", "graph": "PPR", "hybrid": "Hybrid",
             "hybrid_mmr": "Hybrid\n+ MMR"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    colors = [PALETTE["neutral"], PALETTE["red_1"], PALETTE["blue_main"], PALETTE["red_2"], PALETTE["green_3"]]
    bars(axes[0], [names[r["method"]] for r in rows], [r["ndcg_at_10"] for r in rows], colors,
         errors=[r["ndcg_at_10_ci95"] for r in rows])
    axes[0].set_ylabel("nDCG@10")
    for r, color in zip(rows, colors):
        axes[1].scatter(r["diversity"], r["ndcg_at_10"], s=3000 * r["catalog_coverage"] + 60, color=color,
                        edgecolor="black", linewidth=1.2, zorder=3)
        axes[1].annotate(names[r["method"]].replace("\n", " "), (r["diversity"], r["ndcg_at_10"]),
                         xytext=(12, -16) if r["method"] == "hybrid" else (10, 6),
                         textcoords="offset points", fontsize=12)
    axes[1].set(xlabel="Intra-list diversity (1 minus mean cosine)", ylabel="nDCG@10", xlim=(0.25, 1.15), ylim=(0, 0.22))
    save(fig, "holdout")


def corpus(results):
    topics = results["graph"]["topics"]
    short = {"peer review": "PeerRead additions"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].barh(range(len(topics)), list(topics.values()), color=PALETTE["blue_secondary"], edgecolor="black",
                 linewidth=1.2)
    axes[0].set_yticks(range(len(topics)), [short.get(t, t.capitalize()) for t in topics], fontsize=12)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Papers, by first collection query")
    years = {int(y): v for y, v in results["graph"]["years"].items() if int(y) >= 2000}
    axes[1].bar(list(years), list(years.values()), color=PALETTE["blue_secondary"], edgecolor="black", linewidth=0.8,
                width=0.8)
    axes[1].set_xlabel("Publication year")
    axes[1].set_ylabel("Papers")
    save(fig, "corpus")


def latency(results):
    rows = results["latency"]["summary"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(rows))
    for shift, key, color in ((-0.18, "p50_ms", PALETTE["blue_main"]), (0.18, "p95_ms", PALETTE["red_2"])):
        ax.bar(x + shift, [r[key] for r in rows], 0.34, color=color, edgecolor="black", linewidth=1.2,
               label=key.split("_")[0])
    ax.set_xticks(x, [r["method"].upper().replace("_", " ") for r in rows])
    ax.set_ylabel("Milliseconds")
    ax.legend(loc="upper left")
    save(fig, "latency")


def network():
    papers = [json.loads(x) for x in (ROOT / "data/papers.jsonl").open(encoding="utf8") if x.strip()]
    engine = MiningEngine(papers)
    ids = sorted(engine.ids, key=lambda p: -engine.pagerank[p])[:160]
    graph = engine.graph.subgraph(ids)
    pos = nx.spring_layout(graph.to_undirected(), seed=42, iterations=180, k=0.65)
    colors = [PALETTE[c] for c in ("blue_main", "red_strong", "green_3", "teal", "violet", "gray")]
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_edges(graph, pos, ax=ax, width=0.6, alpha=0.2, arrows=True, arrowsize=5, node_size=50)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=[40 + 700 * engine.pagerank[p] / max(engine.pr) for p in graph],
                           node_color=[colors[engine.communities[p] % 6] for p in graph], edgecolors="black",
                           linewidths=0.5)
    for p in ids[:3]:
        title = engine.by_id[p]["title"]
        ax.annotate(title[:42] + ("..." if len(title) > 42 else ""), pos[p], xytext=(9, 9), textcoords="offset points",
                    fontsize=10, bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.axis("off")
    save(fig, "network")


def architecture():
    fig, ax = plt.subplots(figsize=(12, 4.1))
    ax.set_xlim(0, 12)
    ax.set_ylim(0.1, 4.1)
    ax.axis("off")

    def box(x, y, w, h, title, body, key=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.06",
                                    fc=PALETTE["blue_main"] if key else "white", ec="black", lw=1.5))
        color = "white" if key else "black"
        ax.text(x + 0.15, y + h - 0.35, title, fontsize=12, fontweight="bold", color=color)
        ax.text(x + 0.15, y + h - 0.72, body, fontsize=10.5, va="top", linespacing=1.6, color=color)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color="black", lw=1.3))

    box(0.15, 2.5, 2.3, 1.5, "Public sources", "OpenAlex works\nPeerRead reviews")
    box(3.0, 2.5, 2.45, 1.5, "Ingest and clean", "Cached API responses\nDeduplication, hashes")
    box(6.05, 2.5, 2.5, 1.5, "Mining core", "Text retrieval, graphs\nReviews, ranking", key=True)
    box(9.15, 2.5, 2.6, 1.5, "Workspace", "FastAPI and browser UI\nSQLite library")
    box(3.0, 0.2, 2.45, 1.3, "Apache Solr", "Optional Lucene BM25\nDocker profile")
    box(6.05, 0.2, 2.5, 1.3, "Experiments", "Frozen tasks and splits\nPer task scores")
    box(9.15, 0.2, 2.6, 1.3, "Artifacts", "JSON results, figures\nReport and slides")
    for a, b in (((2.45, 3.25), (3.0, 3.25)), ((5.45, 3.25), (6.05, 3.25)), ((8.55, 3.25), (9.15, 3.25)),
                 ((4.2, 2.5), (4.2, 1.5)), ((7.3, 2.5), (7.3, 1.5)), ((8.55, 0.85), (9.15, 0.85))):
        arrow(a, b)
    save(fig, "architecture")


def catalog():
    data = load("catalog_results.json")
    conferences = list(dict.fromkeys(c["conference"] for c in data["coverage"]))
    years = sorted({c["year"] for c in data["coverage"]})
    counts = {(c["conference"], c["year"]): c["papers"] for c in data["coverage"]}
    values = np.array([[counts.get((c, y), 0) for y in years] for c in conferences])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.imshow(values, cmap=BLUES, vmin=0, vmax=values.max() * 1.1, aspect="auto")
    for i in range(len(conferences)):
        for j in range(len(years)):
            ax.text(j, i, f"{values[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if values[i, j] > 0.45 * values.max() else "black")
    ax.set_xticks(range(len(years)), years)
    ax.set_yticks(range(len(conferences)), conferences)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "catalog")


def method_names():
    summary = json.loads((ROOT / "evaluation/decision-study/method_retrieval.json").read_text())["summary"]
    groups = [("scierc_visible", "Method name kept"), ("scierc_masked", "Method name masked")]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (arm, label, color) in enumerate((("reader", "Reader", PALETTE["red_2"]),
                                             ("fusion", "Reader + BM25", PALETTE["blue_main"]))):
        hits = [summary[g]["arms"][arm]["hit3"] for g, _ in groups]
        n = [summary[g]["queries"] for g, _ in groups]
        y = np.array([1, 0]) + (0.17 if i == 0 else -0.17)
        ax.barh(y, [100 * h / q for h, q in zip(hits, n)], 0.32, color=color, edgecolor="black", linewidth=1.2,
                label=label)
        for yy, h, q in zip(y, hits, n):
            ax.text(100 * h / q + 0.8, yy, f"{h}/{q}", va="center", fontsize=12)
    ax.set_yticks([1, 0], [g for _, g in groups])
    ax.set_xlabel("Queries with a right paper in the top 3 (%)")
    ax.set_xlim(0, 62)
    ax.legend(loc="lower right")
    save(fig, "method_names")


if __name__ == "__main__":
    results = load("results.json")
    signals()
    decision()
    rankers()
    cost()
    retrieval_and_reviews(results)
    holdout(results)
    corpus(results)
    latency(results)
    architecture()
    catalog()
    method_names()
    network()
    print("figures written to", OUT)
