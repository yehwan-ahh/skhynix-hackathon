# 반도체 공정 이상탐지 & 원인분석 대시보드

SK하이닉스 AI 해커톤 2026 포트폴리오 프로젝트

## 목표

반도체 에칭(Etch) 공정에서 ER(Etch Rate) 이상이 발생했을 때, 어떤 공정 파라미터가 원인일 확률이 높은지 순위로 짚어주는 설명가능 AI(XAI) 모델을 만들고, 이상탐지부터 원인분석까지의 과정을 Streamlit 대시보드로 시연한다.

- 회사(듀폰) 사내 데이터는 사용하지 않음 — 공개 반도체 공정 데이터셋(SECOM)으로 개념 검증(PoC) 진행
- 이상탐지(분류) + SHAP 기반 원인 파라미터 랭킹 2단계 파이프라인

## 프로젝트 구조

```
skhynix-hackathon/
├── data/           # 원본/전처리 데이터 (git 제외)
├── notebooks/       # 탐색적 분석(EDA), 모델 실험 노트북
├── src/              # 전처리, 모델링, SHAP 분석 모듈
├── app/              # Streamlit 대시보드
├── requirements.txt  # 핵심 의존성
└── requirements-lock.txt  # 전체 고정 버전
```

## 기술 스택

- 데이터 처리: pandas, numpy
- 모델링: scikit-learn, XGBoost, imbalanced-learn(SMOTE)
- 설명가능 AI: SHAP
- 대시보드: Streamlit, plotly

## 데이터셋

UCI/Kaggle SECOM Semiconductor Manufacturing 데이터셋 — 센서 590개 + Pass/Fail 라벨, 약 1,567개 샘플.

## 개발 환경 설정

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 진행 상황

- [x] 개발환경 구성
- [ ] 데이터셋 확보 및 EDA
- [ ] 이상탐지 모델 학습
- [ ] SHAP 원인분석 적용
- [ ] Streamlit 대시보드 구축
