from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "titanic.csv"

st.set_page_config(
    page_title="타이타닉 생존 분석",
    page_icon="🚢",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["성별"] = data["Sex"].map({"female": "여성", "male": "남성"})
    data["객실 등급"] = data["Pclass"].map({1: "1등석", 2: "2등석", 3: "3등석"})
    data["생존 여부"] = data["Survived"].map({1: "생존", 0: "사망"})
    return data


data = load_data()

st.title("🚢 타이타닉 생존 분석 대시보드")
st.markdown(
    "타이타닉 승객 891명의 데이터를 바탕으로 성별과 객실 등급에 따른 "
    "생존 경향을 살펴봅니다. 왼쪽 필터를 바꾸면 모든 지표가 함께 갱신됩니다."
)

st.sidebar.header("분석 필터")
selected_sexes = st.sidebar.multiselect(
    "성별",
    options=["female", "male"],
    default=["female", "male"],
    format_func=lambda value: {"female": "여성", "male": "남성"}[value],
)
selected_classes = st.sidebar.multiselect(
    "객실 등급",
    options=[1, 2, 3],
    default=[1, 2, 3],
    format_func=lambda value: f"{value}등석",
)
minimum_age = int(data["Age"].min())
maximum_age = int(data["Age"].max())
selected_age = st.sidebar.slider(
    "나이 범위",
    min_value=minimum_age,
    max_value=maximum_age,
    value=(minimum_age, maximum_age),
)
include_unknown_age = st.sidebar.checkbox("나이 미상 승객 포함", value=True)

age_condition = data["Age"].between(selected_age[0], selected_age[1])
if include_unknown_age:
    age_condition = age_condition | data["Age"].isna()

filtered = data[
    data["Sex"].isin(selected_sexes)
    & data["Pclass"].isin(selected_classes)
    & age_condition
].copy()

if filtered.empty:
    st.warning("선택한 조건에 해당하는 승객이 없습니다. 왼쪽 필터를 변경해 주세요.")
    st.stop()

passenger_count = len(filtered)
survivor_count = int(filtered["Survived"].sum())
death_count = passenger_count - survivor_count
survival_rate = filtered["Survived"].mean() * 100
average_age = filtered["Age"].mean()

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("승객 수", f"{passenger_count:,}명")
metric2.metric("생존자", f"{survivor_count:,}명")
metric3.metric("생존율", f"{survival_rate:.1f}%")
metric4.metric("평균 나이", f"{average_age:.1f}세")

overview_tab, analysis_tab, images_tab, data_tab = st.tabs(
    ["전체 요약", "성별·객실 분석", "분석 결과 이미지", "데이터 탐색"]
)

with overview_tab:
    st.subheader("생존 현황")
    left, right = st.columns(2)

    with left:
        survival_counts = pd.DataFrame(
            {"인원": [survivor_count, death_count]}, index=["생존", "사망"]
        )
        st.bar_chart(survival_counts)

    with right:
        st.markdown("#### 핵심 분석 결과")
        st.markdown(
            """
            - 전체 승객 891명 중 342명이 생존해 전체 생존율은 **38.4%**입니다.
            - 여성 생존율은 약 **74.2%**, 남성 생존율은 약 **18.9%**입니다.
            - 객실 등급별 생존율은 1등석 **63.0%**, 2등석 **47.3%**, 3등석 **24.2%**입니다.
            - 이 결과는 성별과 객실 등급이 생존 여부와 관련되어 있음을 보여주지만, 직접적인 인과관계를 증명하지는 않습니다.
            """
        )

    st.subheader("필터 적용 데이터 미리보기")
    preview_columns = [
        "PassengerId",
        "Name",
        "성별",
        "Age",
        "객실 등급",
        "Fare",
        "생존 여부",
    ]
    st.dataframe(filtered[preview_columns].head(20), width="stretch", hide_index=True)

with analysis_tab:
    st.subheader("성별에 따른 생존율")
    sex_stats = (
        filtered.groupby("성별", observed=True)["Survived"]
        .agg(승객수="count", 생존자="sum", 생존율="mean")
        .reset_index()
    )
    sex_stats["생존율"] = sex_stats["생존율"] * 100
    st.bar_chart(sex_stats.set_index("성별")["생존율"])
    st.dataframe(
        sex_stats.style.format({"생존율": "{:.1f}%"}),
        width="stretch",
        hide_index=True,
    )

    st.subheader("객실 등급에 따른 생존율")
    class_stats = (
        filtered.groupby(["Pclass", "객실 등급"], observed=True)["Survived"]
        .agg(승객수="count", 생존자="sum", 생존율="mean")
        .reset_index()
        .sort_values("Pclass")
    )
    class_stats["생존율"] = class_stats["생존율"] * 100
    st.bar_chart(class_stats.set_index("객실 등급")["생존율"])
    st.dataframe(
        class_stats.drop(columns="Pclass").style.format({"생존율": "{:.1f}%"}),
        width="stretch",
        hide_index=True,
    )

    st.subheader("성별과 객실 등급을 함께 비교")
    interaction = filtered.pivot_table(
        index="객실 등급",
        columns="성별",
        values="Survived",
        aggfunc="mean",
    )
    interaction = interaction.reindex(["1등석", "2등석", "3등석"]) * 100
    st.bar_chart(interaction)
    st.dataframe(interaction.style.format("{:.1f}%"), width="stretch")

with images_tab:
    st.subheader("기존 분석 결과")
    image_items = [
        ("핵심 최종 가설", "핵심_최종가설_요약.png"),
        ("가설 1: 성별 생존율", "가설1_성별_생존율.png"),
        ("가설 2: 객실 등급별 생존율", "가설2_객실등급별_생존율.png"),
        ("가설 3: 성별과 객실 등급", "가설3_성별과_객실등급별_생존율.png"),
        ("통계적 가설 비교", "통계적가설_H0_H1_비교.png"),
        ("성별·객실 등급별 생존율", "성별_객실등급별_생존율_막대그래프.png"),
    ]

    for start in range(0, len(image_items), 2):
        columns = st.columns(2)
        for column, (caption, filename) in zip(columns, image_items[start : start + 2]):
            image_path = APP_DIR / filename
            with column:
                if image_path.is_file():
                    st.image(str(image_path), caption=caption, width="stretch")
                else:
                    st.warning(f"이미지를 찾을 수 없습니다: {filename}")

with data_tab:
    st.subheader("승객 데이터 탐색")
    search_text = st.text_input("승객 이름 검색")
    displayed = filtered
    if search_text:
        displayed = displayed[
            displayed["Name"].str.contains(search_text, case=False, na=False)
        ]

    display_columns = [
        "PassengerId",
        "Name",
        "성별",
        "Age",
        "객실 등급",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "생존 여부",
    ]
    st.dataframe(displayed[display_columns], width="stretch", hide_index=True)
    st.download_button(
        "필터 결과 CSV 다운로드",
        data=displayed[display_columns].to_csv(index=False).encode("utf-8-sig"),
        file_name="titanic_filtered.csv",
        mime="text/csv",
    )

st.caption("데이터 출처: Kaggle Titanic 데이터셋 · 분석 결과는 교육 목적으로 제공됩니다.")

