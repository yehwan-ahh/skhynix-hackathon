"""
반도체 공정 이상탐지 & 원인분석 대시보드
SK하이닉스 AI 해커톤 2026 포트폴리오

실행 방법:
    streamlit run app.py
(이 파일이 있는 app 폴더 안에서, venv가 활성화된 상태로 실행하세요)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

APP_DIR = Path(__file__).parent

st.set_page_config(
    page_title="반도체 공정 이상탐지 & 원인분석",
    layout="wide",
)

# ── 데이터/모델 로드 (캐시로 한 번만 로드) ──────────────────────────────


@st.cache_resource
def load_model_bundle():
    import joblib

    return joblib.load(APP_DIR / "model.pkl")


@st.cache_data
def load_sample_data():
    X = pd.read_csv(APP_DIR / "X_test_sample.csv")
    y = pd.read_csv(APP_DIR / "y_test_sample.csv").squeeze("columns")
    return X, y


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def compute_global_importance(_explainer, X: pd.DataFrame) -> pd.DataFrame:
    shap_vals = _explainer.shap_values(X)
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
        .iloc[::-1]
    )


try:
    bundle = load_model_bundle()
    model = bundle["model"]
    threshold = bundle["threshold"]
    feature_names = bundle["feature_names"]
    X_test, y_test = load_sample_data()
    explainer = get_explainer(model)
except FileNotFoundError:
    st.error(
        "model.pkl / X_test_sample.csv / y_test_sample.csv 파일을 찾을 수 없어요. "
        "노트북(01_eda.ipynb)의 마지막 셀(모델 저장)을 먼저 실행했는지 확인해주세요."
    )
    st.stop()

# ── 헤더 ────────────────────────────────────────────────────────────

st.title("반도체 공정 이상탐지 & 원인분석 대시보드")
st.caption("SK하이닉스 AI 해커톤 2026 포트폴리오 — SECOM 공개 데이터셋 기반 개념 검증(PoC)")

with st.expander("이 대시보드에 대해 (먼저 읽어주세요)"):
    st.markdown(
        f"""
- **데이터**: 실제 사내 데이터가 아니라, UCI Machine Learning Repository의 공개
  반도체 제조 공정 데이터셋(**SECOM**)을 사용한 개념 검증(PoC)입니다. 컬럼명이
  숫자인 것은 SECOM 데이터셋 자체가 기업 기밀 보호를 위해 센서명을 익명화했기
  때문입니다.
- **모델**: XGBoost 분류기 + SMOTE(소수 클래스 오버샘플링). 판정 임계값은
  **{threshold}** 로 설정했습니다 (불량 비율이 6.6%로 낮은 데이터라 recall을
  우선하기 위해 기본값 0.5보다 낮게 조정).
- **설명(원인분석)**: SHAP(SHapley Additive exPlanations)으로 각 판정에 대해
  어떤 피처(센서)가 얼마나, 어느 방향으로 기여했는지 계산합니다.
- **한계**: 불량 신호 자체가 약한 데이터라 재현율이 높지 않습니다. 실제 fab
  데이터라면 공정 도메인 지식을 반영한 feature selection으로 더 개선할 수
  있습니다.
"""
    )

st.divider()

# ── 샘플 선택 + 판정 결과 ──────────────────────────────────────────

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("샘플 선택")
    idx = st.selectbox(
        "test셋 샘플 인덱스를 선택하세요",
        options=list(range(len(X_test))),
        index=0,
        help="SECOM test셋(314개 샘플) 중 하나를 골라 예측/원인 분석을 확인합니다.",
    )
    sample = X_test.iloc[[idx]]
    actual = y_test.iloc[idx]

    proba = float(model.predict_proba(sample)[0, 1])
    pred = int(proba >= threshold)

    st.metric("예측 불량 확률", f"{proba:.1%}")
    if pred == 1:
        st.error("⚠️ 불량(이상) 의심")
    else:
        st.success("✅ 정상")

    actual_label = "불량" if actual == 1 else "정상"
    match = "일치" if pred == actual else "불일치 (오판)"
    st.caption(f"실제 라벨: **{actual_label}** · 판정 임계값: {threshold} · 예측-실제 {match}")

# ── SHAP 원인 파라미터 랭킹 ────────────────────────────────────────

with col2:
    st.subheader("이 샘플의 원인 파라미터 Top-10")
    shap_values = explainer.shap_values(sample)
    row_shap = shap_values[0]

    ranking = pd.DataFrame({"feature": feature_names, "shap_value": row_shap})
    ranking["abs_shap"] = ranking["shap_value"].abs()
    top10 = ranking.sort_values("abs_shap", ascending=False).head(10).iloc[::-1]

    # 진단(diverging) 컬러: 양수(불량 방향)=레드, 음수(정상 방향)=블루
    colors = ["#b2182b" if v > 0 else "#2166ac" for v in top10["shap_value"]]

    fig = go.Figure(
        go.Bar(
            x=top10["shap_value"],
            y=top10["feature"].astype(str),
            orientation="h",
            marker_color=colors,
            hovertemplate="피처 %{y}<br>SHAP 기여도: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="SHAP 기여도 (음수=정상 방향 ← · → 양수=불량 방향)",
        yaxis_title="센서(피처) ID",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── 전체 데이터 기준 글로벌 중요도 ─────────────────────────────────

st.subheader("전체 test셋 기준 주요 피처 (Global Importance)")
st.caption("개별 샘플이 아니라, 전체 데이터에서 평균적으로 판정에 가장 큰 영향을 준 센서들이에요.")

global_imp = compute_global_importance(explainer, X_test)

fig2 = go.Figure(
    go.Bar(
        x=global_imp["mean_abs_shap"],
        y=global_imp["feature"].astype(str),
        orientation="h",
        marker_color="#4393c3",
        hovertemplate="피처 %{y}<br>평균 |SHAP|: %{x:.4f}<extra></extra>",
    )
)
fig2.update_layout(
    xaxis_title="평균 |SHAP| (전체 판정에 대한 평균 영향력)",
    yaxis_title="센서(피처) ID",
    height=450,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig2, use_container_width=True)


