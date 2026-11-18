"""Draw the paired test-query differences from the saved evaluation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "10010426": "Lập luận trong hội thoại",
    "1198964": "Phân tích cú pháp phụ thuộc",
    "53080736": "Phân loại quan hệ FewRel",
    "1936997": "Ngữ nghĩa của phủ định",
    "174799296": "Cú pháp trong Transformer",
    "5052952": "Đánh giá tính xác thực",
    "80628431": "Thiên lệch quy nạp của RNN",
    "53082542": "Dịch máy cá nhân hóa",
    "102353905": "Trích quan hệ cấp văn bản",
}


def main():
    result = json.loads(
        (ROOT / "evaluation/decision-study/decision_comparison.json").read_text()
    )
    models = result["models"]
    laya = models["combined"]["external_test"]["ndcg_by_query"]
    text = models["text_only"]["external_test"]["ndcg_by_query"]
    rows = sorted(((laya[q] - text[q], LABELS[q]) for q in laya), reverse=True)
    fig, ax = plt.subplots(figsize=(7.3, 4.15), layout="constrained")
    for i, (delta, label) in enumerate(rows):
        ax.barh(i, delta, height=0.55, color="#bd2636" if delta > 0 else "#607585")
        ax.text(
            delta + (0.003 if delta >= 0 else -0.003),
            i,
            f"{delta:+.3f}".replace(".", ","),
            ha="left" if delta >= 0 else "right",
            va="center",
            fontsize=10,
        )
    ax.axvline(0, color="#253746", lw=0.8)
    ax.set_xlim(-0.133, 0.117)
    ax.set_yticks(range(len(rows)), labels=[label for _, label in rows])
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlabel("Thay đổi nDCG@10 khi thêm hai điểm của Laya")
    ax.tick_params(axis="y", length=0)
    ax.spines[:].set_visible(False)
    fig.savefig(ROOT / "artifacts/figures/decision_by_query.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
