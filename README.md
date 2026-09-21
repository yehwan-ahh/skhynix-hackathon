# 반도체 공정 이상탐지 & 원인분석 대시보드

SK하이닉스 AI 해커톤 2026 포트폴리오 프로젝트

## 문제 정의

반도체 에칭(Etch) 공정에서 ER(Etch Rate) 같은 결과 지표에 이상이 발생했을 때, 담당자는 "어떤 공정 파라미터가 문제였을까?"를 경험과 감으로 추적하는 경우가 많다. 이 프로젝트는 이 과정을 데이터 기반으로 보조하는 것을 목표로 한다:

1. 공정 결과가 정상인지 이상인지 자동으로 판정하고 (**이상탐지**)
2. 이상으로 판정됐다면 어떤 파라미터가 얼마나 영향을 미쳤는지 순위로 보여준다 (**원인분석**)

## 접근 방법

- **데이터**: 회사(듀폰) 사내 데이터는 일절 사용하지 않았다. 대신 구조가 유사한 공개 반도체 제조 공정 데이터셋(**UCI SECOM**)으로 개념 검증(PoC)을 진행했다.
- **전처리**: 결측치 50% 이상 컬럼 제거 → 라벨 인코딩(-1/1 → 0/1) → 잔여 결측치 중앙값 대체 → 분산 0인 상수 컬럼 제거 → train/test 8:2 분리(불량 비율 유지, stratify).
- **이상탐지 모델**: XGBoost 분류기 + SMOTE(소수 클래스 오버샘플링). 불량 비율이 6.6%로 낮아 판정 임계값을 기본값 0.5 대신 **0.3**으로 낮춰 재현율(recall)을 우선했다.
- **원인분석**: SHAP(TreeExplainer)으로 개별 판정에 대한 피처별 기여도를 계산하고, 절댓값 기준 상위 파라미터를 순위로 제시한다.
- **시연**: Streamlit + Plotly로 샘플 선택 → 판정 결과 → 원인 파라미터 Top-10 → 전체 데이터 기준 주요 피처를 한 화면에서 볼 수 있는 대시보드를 구축했다.

## 결과

| 지표 | 값 |
|---|---|
| AUC | 0.69 |
| 불량(Fail) precision | 0.33 |
| 불량(Fail) recall | 0.19 |
| 불량(Fail) F1 | 0.24 |
| 판정 임계값 | 0.3 |

원본 신호가 약한 공개 데이터셋 특성상 재현율이 높지 않지만, AUC 0.69는 무작위(0.5)보다 뚜렷한 신호가 있다는 것을 보여준다. 이 프로젝트의 핵심은 "성능 수치 자체"보다 **이상탐지 → 원인분석 → 시각화로 이어지는 end-to-end 파이프라인**을 만들고, AI 코딩 도구와 함께 처음부터 구현·검증했다는 데 있다.

## 한계와 실제 현업과의 차이

- SECOM은 익명화된 범용 반도체 공정 데이터셋으로, 실제 에칭 ER 데이터가 아니다.
- 결측치·저분산 컬럼 제거를 순수 통계 기준으로만 했다. 실제 fab에서는 공정 엔지니어와 협의해 "왜 결측인지"부터 확인하고, SPC 관리도(Cpk, UCL/LCL) 기준으로 판단한다.
- train/test를 무작위로 분리했다. 실제 공정은 시간에 따라 드리프트가 있어, 현업에서는 시간 기준으로 분리해 data leakage를 방지한다.
- 이 모델은 재현율이 낮아 실제 배포에는 추가 개선(도메인 지식 기반 feature selection, 임계값 재튜닝 등)이 필요하다.

## 프로젝트 구조

```
skhynix-hackathon/
├── data/              # 원본/전처리 데이터 (git 제외)
├── notebooks/         # 01_eda.ipynb — EDA, 전처리, 모델 학습, SHAP 분석 전 과정
├── src/               # 전처리, 모델링, SHAP 분석 모듈
├── app/               # Streamlit 대시보드 (app.py) + 학습된 모델/샘플 데이터
├── requirements.txt   # 핵심 의존성
└── requirements-lock.txt  # 전체 고정 버전
```

## 기술 스택

- 데이터 처리: pandas, numpy
- 모델링: scikit-learn, XGBoost, imbalanced-learn(SMOTE)
- 설명가능 AI: SHAP
- 대시보드: Streamlit, plotly
- 구현 방식: AI 코딩 도구(Claude)와 함께 설계부터 디버깅까지 진행, 전 과정을 직접 이해하고 검증

## 실행 방법

**환경 설정**
```bash
python -m venv venv
# Windows
venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

**노트북 재현** (`notebooks/01_eda.ipynb`): 데이터 로드부터 모델 저장까지 순서대로 실행

**대시보드 실행**
```bash
cd app
streamlit run app.py
```

## 데이터셋

UCI Machine Learning Repository — [SECOM (SEmiCOnductor Manufacturing)](https://archive.ics.uci.edu/ml/datasets/SECOM). 센서 590개 + Pass/Fail 라벨, 약 1,567개 샘플, 불량 비율 약 6.6%.

## 진행 상황

- [x] 개발환경 구성
- [x] 데이터셋 확보 및 EDA
- [x] 이상탐지 모델 학습
- [x] SHAP 원인분석 적용
- [x] Streamlit 대시보드 구축
