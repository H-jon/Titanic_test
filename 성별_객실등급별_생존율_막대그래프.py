from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "titanic.csv"
OUTPUT_PATH = BASE_DIR / "성별_객실등급별_생존율_막대그래프.png"


def main() -> None:
    data = pd.read_csv(CSV_PATH)

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    classes = [1, 2, 3]
    class_labels = ["1등실", "2등실", "3등실"]
    male_rates = [
        data.loc[(data["Sex"] == "male") & (data["Pclass"] == pclass), "Survived"].mean() * 100
        for pclass in classes
    ]
    female_rates = [
        data.loc[(data["Sex"] == "female") & (data["Pclass"] == pclass), "Survived"].mean() * 100
        for pclass in classes
    ]

    x = np.arange(len(classes))
    width = 0.34
    male_color = "#3B82F6"
    female_color = "#A855F7"

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("#F8FAFC")

    male_bars = ax.bar(
        x - width / 2,
        male_rates,
        width,
        label="남성",
        color=male_color,
        edgecolor="white",
        linewidth=1.5,
    )
    female_bars = ax.bar(
        x + width / 2,
        female_rates,
        width,
        label="여성",
        color=female_color,
        edgecolor="white",
        linewidth=1.5,
    )

    ax.bar_label(male_bars, fmt="%.1f%%", padding=5, fontsize=12, weight="bold")
    ax.bar_label(female_bars, fmt="%.1f%%", padding=5, fontsize=12, weight="bold")

    ax.set_title(
        "타이타닉 성별·객실 등급별 생존율",
        fontsize=22,
        weight="bold",
        color="#111827",
        pad=28,
    )
    ax.text(
        0.5,
        1.015,
        "가설: 여성일수록, 객실 등급이 높을수록 생존율이 높았을 것이다",
        transform=ax.transAxes,
        ha="center",
        fontsize=12.5,
        color="#475569",
        weight="bold",
    )
    ax.set_ylabel("생존율(%)", fontsize=13, weight="bold")
    ax.set_xlabel("객실 등급", fontsize=13, weight="bold", labelpad=12)
    ax.set_xticks(x, class_labels, fontsize=13, weight="bold")
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=12,
        ncol=2,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"막대그래프를 저장했습니다: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
