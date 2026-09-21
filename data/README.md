# 데이터

`uci-secom.csv`는 git에 커밋하지 않음 (용량 + 재현 목적). 아래 명령으로 받는다:

```bash
curl -sSL -o data/uci-secom.csv "https://raw.githubusercontent.com/sharmaroshan/SECOM-Detecting-Defected-Items/master/uci-secom.csv"
```

## 데이터 요약 (2026-09-21 확인)

- shape: (1567, 592) — Time 컬럼 1개 + 센서 컬럼 590개 + 라벨 컬럼(`Pass/Fail`) 1개
- 라벨: `-1` = Pass (1463건), `1` = Fail (104건, 약 6.6%) — 불균형 데이터
- 결측치: 전체 평균 약 4.5%. 일부 컬럼(158, 292, 293, 157 등)은 결측치 비율 91%+ → 전처리 단계에서 제거 검토 필요

원본 출처: [UCI Machine Learning Repository — SECOM](https://archive.ics.uci.edu/ml/datasets/SECOM) (GitHub 미러: [sharmaroshan/SECOM-Detecting-Defected-Items](https://github.com/sharmaroshan/SECOM-Detecting-Defected-Items))
