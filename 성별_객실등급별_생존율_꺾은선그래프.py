from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic.csv"
OUTPUT_PATH = BASE_DIR / "성별_객실등급별_생존율_꺾은선그래프.png"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    classes = [1, 2, 3]
    labels = ["1등실", "2등실", "3등실"]
    male = [
        data.loc[(data["Sex"] == "male") & (data["Pclass"] == pclass), "Survived"].mean() * 100
        for pclass in classes
    ]
    female = [
        data.loc[(data["Sex"] == "female") & (data["Pclass"] == pclass), "Survived"].mean() * 100
        for pclass in classes
    ]
    overall = [
        data.loc[data["Pclass"] == pclass, "Survived"].mean() * 100
        for pclass in classes
    ]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("#F8FAFC")

    lines = [
        (male, "남성", "#3B82F6", "o", "-"),
        (female, "여성", "#A855F7", "o", "-"),
        (overall, "전체", "#475569", "s", "--"),
    ]
    for values, label, color, marker, linestyle in lines:
        ax.plot(
            labels,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=3,
            markersize=9,
        )
        for index, value in enumerate(values):
            offset = 4 if label != "전체" else -7
            ax.annotate(
                f"{value:.1f}%",
                (index, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if offset > 0 else "top",
                fontsize=11.5,
                color=color,
                weight="bold",
            )

    ax.set_title(
        "객실 등급에 따른 성별 생존율 변화",
        fontsize=22,
        weight="bold",
        color="#111827",
        pad=24,
    )
    ax.text(
        0.5,
        1.01,
        "1등실에서 3등실로 갈수록 생존율이 낮아지는 경향",
        transform=ax.transAxes,
        ha="center",
        fontsize=12.5,
        color="#475569",
        weight="bold",
    )
    ax.set_xlabel("객실 등급", fontsize=13, weight="bold", labelpad=12)
    ax.set_ylabel("생존율(%)", fontsize=13, weight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", color="#CBD5E1", alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=12)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"꺾은선 그래프를 저장했습니다: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
