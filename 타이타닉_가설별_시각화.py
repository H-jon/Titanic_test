from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA = pd.read_csv(BASE_DIR / "titanic.csv")

MALE = "#3B82F6"
FEMALE = "#A855F7"
CLASS_COLORS = ["#0F766E", "#14B8A6", "#99F6E4"]
TEXT = "#111827"
GRID = "#CBD5E1"


def setup_korean_font() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#F8FAFC")
    ax.grid(axis="y", linestyle="--", linewidth=1, alpha=0.45, color=GRID)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("생존율(%)", fontsize=12, weight="bold")


def add_labels(ax: plt.Axes, bars) -> None:
    ax.bar_label(bars, fmt="%.1f%%", padding=5, fontsize=13, weight="bold", color=TEXT)


def save_sex_hypothesis() -> None:
    rates = DATA.groupby("Sex")["Survived"].mean().reindex(["male", "female"]) * 100
    counts = DATA["Sex"].value_counts().reindex(["male", "female"])

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    bars = ax.bar(["남성", "여성"], rates, color=[MALE, FEMALE], width=0.55)
    style_axis(ax)
    add_labels(ax, bars)
    ax.set_title("가설 1: 여성의 생존율이 남성보다 높을 것이다", fontsize=19, weight="bold", pad=20)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            3,
            f"전체 {count}명",
            ha="center",
            fontsize=11,
            color="white",
            weight="bold",
        )
    fig.tight_layout()
    fig.savefig(BASE_DIR / "가설1_성별_생존율.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_class_hypothesis() -> None:
    rates = DATA.groupby("Pclass")["Survived"].mean().reindex([1, 2, 3]) * 100
    counts = DATA["Pclass"].value_counts().reindex([1, 2, 3])

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    bars = ax.bar(["1등실", "2등실", "3등실"], rates, color=CLASS_COLORS, width=0.6)
    style_axis(ax)
    add_labels(ax, bars)
    ax.set_title("가설 2: 객실 등급이 높을수록 생존율이 높을 것이다", fontsize=19, weight="bold", pad=20)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            3,
            f"전체 {count}명",
            ha="center",
            fontsize=11,
            color="white" if bar.get_height() > 30 else TEXT,
            weight="bold",
        )
    fig.tight_layout()
    fig.savefig(BASE_DIR / "가설2_객실등급별_생존율.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_interaction_hypothesis() -> None:
    classes = [1, 2, 3]
    male_rates = [
        DATA.loc[(DATA["Sex"] == "male") & (DATA["Pclass"] == pclass), "Survived"].mean() * 100
        for pclass in classes
    ]
    female_rates = [
        DATA.loc[(DATA["Sex"] == "female") & (DATA["Pclass"] == pclass), "Survived"].mean() * 100
        for pclass in classes
    ]

    x = np.arange(3)
    width = 0.34
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    male_bars = ax.bar(x - width / 2, male_rates, width, label="남성", color=MALE)
    female_bars = ax.bar(x + width / 2, female_rates, width, label="여성", color=FEMALE)
    style_axis(ax)
    add_labels(ax, male_bars)
    add_labels(ax, female_bars)
    ax.set_xticks(x, ["1등실", "2등실", "3등실"], fontsize=12, weight="bold")
    ax.set_title("가설 3: 성별과 객실 등급이 함께 생존과 관련 있을 것이다", fontsize=19, weight="bold", pad=20)
    ax.legend(frameon=False, fontsize=12, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "가설3_성별과_객실등급별_생존율.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_core_hypothesis_summary() -> None:
    sex_rates = DATA.groupby("Sex")["Survived"].mean().reindex(["male", "female"]) * 100
    class_rates = DATA.groupby("Pclass")["Survived"].mean().reindex([1, 2, 3]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

    sex_bars = axes[0].bar(["남성", "여성"], sex_rates, color=[MALE, FEMALE], width=0.58)
    style_axis(axes[0])
    add_labels(axes[0], sex_bars)
    axes[0].set_title("여성일수록", fontsize=17, weight="bold", pad=14)
    axes[0].annotate(
        "생존율 +55.3%p",
        xy=(1, sex_rates.iloc[1]),
        xytext=(0.48, 92),
        arrowprops={"arrowstyle": "->", "color": FEMALE, "lw": 2},
        fontsize=12,
        weight="bold",
        color=FEMALE,
    )

    class_bars = axes[1].bar(
        ["1등실", "2등실", "3등실"],
        class_rates,
        color=CLASS_COLORS,
        width=0.62,
    )
    style_axis(axes[1])
    add_labels(axes[1], class_bars)
    axes[1].set_title("객실 등급이 높을수록", fontsize=17, weight="bold", pad=14)
    axes[1].annotate(
        "1등실이 3등실보다 +38.7%p",
        xy=(0, class_rates.iloc[0]),
        xytext=(0.65, 84),
        arrowprops={"arrowstyle": "->", "color": CLASS_COLORS[0], "lw": 2},
        fontsize=11.5,
        weight="bold",
        color=CLASS_COLORS[0],
        ha="center",
    )

    fig.suptitle(
        "핵심·최종 가설: 여성이고 객실 등급이 높을수록 생존율이 높았을 것이다",
        fontsize=21,
        weight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(BASE_DIR / "핵심_최종가설_요약.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_statistical_hypotheses() -> None:
    overall_rate = DATA["Survived"].mean() * 100
    observed = [
        DATA.loc[DATA["Sex"] == "male", "Survived"].mean() * 100,
        DATA.loc[DATA["Sex"] == "female", "Survived"].mean() * 100,
        DATA.loc[DATA["Pclass"] == 1, "Survived"].mean() * 100,
        DATA.loc[DATA["Pclass"] == 2, "Survived"].mean() * 100,
        DATA.loc[DATA["Pclass"] == 3, "Survived"].mean() * 100,
    ]
    labels = ["남성", "여성", "1등실", "2등실", "3등실"]
    colors = [MALE, FEMALE, *CLASS_COLORS]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")

    null_bars = axes[0].bar(labels, [overall_rate] * 5, color="#94A3B8", width=0.65)
    style_axis(axes[0])
    add_labels(axes[0], null_bars)
    axes[0].set_title("귀무가설 H0: 집단별 차이가 없다", fontsize=17, weight="bold", pad=14)
    axes[0].text(
        0.5,
        0.92,
        "비교 기준: 모든 집단이 전체 생존율 38.4%와 같다고 가정",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=10.5,
        color="#475569",
        weight="bold",
    )

    alternative_bars = axes[1].bar(labels, observed, color=colors, width=0.65)
    style_axis(axes[1])
    add_labels(axes[1], alternative_bars)
    axes[1].set_title("대립가설 H1: 집단별 차이가 있다", fontsize=17, weight="bold", pad=14)
    axes[1].text(
        0.5,
        0.92,
        "실제 관찰값: 성별·객실 등급에 따라 생존율이 다름",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=10.5,
        color="#475569",
        weight="bold",
    )

    fig.suptitle("통계적 가설의 시각적 비교", fontsize=22, weight="bold", y=1.02)
    fig.text(
        0.5,
        -0.01,
        "※ H0 그래프는 실제 데이터가 아니라 ‘차이가 없다’는 가정을 시각화한 비교 기준입니다.",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(BASE_DIR / "통계적가설_H0_H1_비교.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_korean_font()
    save_sex_hypothesis()
    save_class_hypothesis()
    save_interaction_hypothesis()
    save_core_hypothesis_summary()
    save_statistical_hypotheses()
    print("모든 가설 시각화를 저장했습니다.")


if __name__ == "__main__":
    main()
