from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


# 이 Python 파일과 같은 폴더에 있는 CSV를 읽습니다.
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "titanic.csv"
OUTPUT_PATH = BASE_DIR / "성별_객실등급별_생존현황_6개그래프.png"


def set_korean_font() -> None:
    """Windows에서 한글이 깨지지 않도록 글꼴을 설정합니다."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    data = pd.read_csv(CSV_PATH)

    # Survived 값: 0은 사망, 1은 생존입니다.
    sex_groups = [("male", "남성"), ("female", "여성")]
    class_groups = [(1, "1등실"), (2, "2등실"), (3, "3등실")]
    # 결과의 의미가 성별과 관계없이 동일하게 읽히도록 색상을 통일합니다.
    # 사망=주황, 생존=청록의 고대비 조합입니다.
    outcome_colors = ["#E76F51", "#2A9D8F"]
    row_label_colors = {"male": "#264653", "female": "#7B2CBF"}

    set_korean_font()
    background_color = "#FFFFFF"
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=background_color)

    group_counts = {}
    for row, (sex_value, sex_label) in enumerate(sex_groups):
        for col, (class_value, class_label) in enumerate(class_groups):
            ax = axes[row, col]
            ax.set_facecolor(background_color)
            chart_colors = outcome_colors
            counts = (
                data.loc[
                    (data["Sex"] == sex_value) & (data["Pclass"] == class_value),
                    "Survived",
                ]
                .value_counts()
                .reindex([0, 1], fill_value=0)
            )
            group_counts[(sex_label, class_label)] = counts
            # 여성 3등실의 50:50 차트는 좌우로 나뉘도록 세로 경계를 사용합니다.
            start_angle = 90 if (sex_value == "female" and class_value == 3) else 0

            _, _, percentage_texts = ax.pie(
                counts,
                colors=chart_colors,
                autopct="%1.1f%%",
                startangle=start_angle,
                counterclock=False,
                pctdistance=1.16,
                wedgeprops={"edgecolor": "white", "linewidth": 3},
                textprops={"fontsize": 14, "color": "#111827", "weight": "bold"},
            )
            for percentage_text in percentage_texts:
                x, _ = percentage_text.get_position()
                if x > 0.15:
                    percentage_text.set_horizontalalignment("left")
                elif x < -0.15:
                    percentage_text.set_horizontalalignment("right")
                else:
                    percentage_text.set_horizontalalignment("center")
            ax.set_title(
                f"전체 {counts.sum()}명\n사망 {counts[0]}명  ·  생존 {counts[1]}명",
                fontsize=12.5,
                weight="bold",
                color="#111827",
                pad=9,
                linespacing=1.35,
            )
            ax.axis("equal")

    fig.suptitle(
        "타이타닉 성별·객실 등급별 생존 현황",
        fontsize=23,
        weight="bold",
        color="#111827",
        y=0.96,
    )
    fig.text(
        0.5,
        0.90,
        "성별 및 객실 등급에 따른 사망·생존 인원과 비율",
        ha="center",
        fontsize=13,
        weight="bold",
        color="#374151",
    )
    for x, (_, class_label) in zip([0.22, 0.50, 0.78], class_groups):
        fig.text(
            x,
            0.82,
            class_label,
            ha="center",
            fontsize=17,
            weight="bold",
            color="#111827",
        )
    fig.text(
        0.055,
        0.60,
        "남성",
        ha="center",
        va="center",
        fontsize=17,
        weight="bold",
        color=row_label_colors["male"],
    )
    fig.text(
        0.055,
        0.31,
        "여성",
        ha="center",
        va="center",
        fontsize=17,
        weight="bold",
        color=row_label_colors["female"],
    )
    legend_items = [
        Patch(facecolor=outcome_colors[0], label="사망"),
        Patch(facecolor=outcome_colors[1], label="생존"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=2,
        fontsize=14,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        handlelength=1.8,
        columnspacing=2.5,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.96,
        bottom=0.12,
        top=0.72,
        wspace=0.24,
        hspace=0.32,
    )
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"파이차트를 저장했습니다: {OUTPUT_PATH}")
    for (sex_label, class_label), counts in group_counts.items():
        print(
            f"{sex_label} {class_label} - "
            f"사망: {counts[0]}명, 생존: {counts[1]}명"
        )


if __name__ == "__main__":
    main()
