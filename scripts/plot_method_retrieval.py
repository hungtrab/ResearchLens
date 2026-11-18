"""Plot the paired named-method sensitivity study from its saved result."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
summary = json.loads(
    (root / "evaluation/decision-study/method_retrieval.json").read_bytes()
)["summary"]
fig, ax = plt.subplots(figsize=(6.6, 3.4), layout="constrained")
for i, (arm, label, color) in enumerate(
    [("reader", "Reader", "#405662"), ("fusion", "Reader + BM25", "#B71E3C")]
):
    groups = [summary[k] for k in ("scierc_visible", "scierc_masked")]
    hits = [g["arms"][arm]["hit3"] for g in groups]
    y = np.array([1, 0]) + (0.15 if i == 0 else -0.15)
    bars = ax.barh(
        y,
        [100 * h / g["queries"] for h, g in zip(hits, groups)],
        height=0.25,
        color=color,
        label=label,
    )
    ax.bar_label(
        bars,
        labels=[f"{n}/{g['queries']}" for n, g in zip(hits, groups)],
        padding=4,
        fontsize=11,
    )
ax.set(
    yticks=[0, 1],
    yticklabels=["Che tên kỹ thuật", "Giữ tên kỹ thuật"],
    xlim=(0, 63),
    ylim=(-0.52, 1.52),
    xticks=[0, 10, 20, 30, 40, 50],
    xlabel="Truy vấn có bài đúng trong top 3 (%)",
)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower right", frameon=False, fontsize=10)
fig.savefig(
    root / "artifacts/figures/method_name_ablation.pdf", metadata={"CreationDate": None}
)
