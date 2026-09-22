from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "titanic.csv"
OUTPUT_PATH = BASE_DIR / "성별_생존여부_파이차트.png"


def main() -> None:
    data = pd.read_csv(CSV_PATH)

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    labels = ["사망", "생존"]
    # 일반 차트에서 자주 쓰이는 고대비 색상: 사망=빨강, 생존=파랑
    colors = ["#E15759", "#4E79A7"]
    sex_groups = [("male", "남성"), ("female", "여성")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, (sex_value, sex_label) in zip(axes, sex_groups):
        counts = (
            data.loc[data["Sex"] == sex_value, "Survived"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        ax.pie(
            counts,
            labels=labels,
            colors=colors,
            autopct=lambda percent, total=counts.sum(): (
                f"{percent:.1f}%\n({percent / 100 * total:.0f}명)"
            ),
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 3},
            textprops={"fontsize": 12, "weight": "bold"},
        )
        ax.set_title(f"{sex_label} 승객", fontsize=15, pad=15)
        ax.axis("equal")

    fig.suptitle("타이타닉 성별 생존 여부", fontsize=18)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"두 개의 파이차트를 저장했습니다: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
