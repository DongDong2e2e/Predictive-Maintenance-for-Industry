# 발전소 설비 고장 예측 프로젝트 (Portfolio)

## 프로젝트 목표
기계 장비의 센서 데이터를 분석하여 잠재적인 고장을 예측하는 머신러닝 모델을 개발하고, 이를 실제 사용 가능한 애플리케이션 형태로 구현합니다. 본 프로젝트는 선제적 유지보수 조치를 통해 설비 안정성을 높이고, 예기치 않은 가동 중단으로 인한 손실을 최소화하는 실용적인 솔루션을 제시합니다.

## 프로젝트 동기
발전소와 같은 대규모 산업 설비에서 예기치 못한 고장은 막대한 경제적 손실과 안전 문제로 이어질 수 있습니다. 본 프로젝트는 데이터 기반의 예측 유지보수 기술이 어떻게 발전소의 안정적이고 효율적인 운영에 기여할 수 있는지 보여주기 위해 시작되었습니다. AI/ML 기술을 활용하여 설비의 상태를 진단하고, 고장 가능성을 사전에 예측하는 역량을 실제 운영 환경에 적용하고자 하는 열망을 담고 있습니다.

## 프로젝트 구조 및 동작 방식

본 프로젝트는 **분석/실험**과 **훈련/예측 애플리케이션**의 두 가지 주요 부분으로 구조화되어 있습니다. 역할 기반 디렉토리 구조를 채택하여 코드, 데이터, 실험 노트북, 모델 결과물을 명확히 분리합니다.

```
.
├── data/
│   └── ai4i2020.csv          # 원본 데이터셋
├── models/
│   └── (비어있음, train.py 실행 후 채워짐)
├── notebooks/
│   └── app.ipynb             # 1. 데이터 탐색, 모델링 실험 및 심화 분석
├── src/
│   ├── __init__.py
│   ├── train.py              # 2. 최적 모델 훈련 및 저장 스크립트
│   └── predict.py            # 3. 실시간 예측 CLI 애플리케이션
├── .gitignore
└── requirements.txt
```

**동작 워크플로우:**
1.  **`notebooks/app.ipynb`**: 데이터의 특성을 파악하고, 다양한 모델링 기법(하이퍼파라미터 튜닝 등)을 실험하여 최적의 모델과 파라미터를 탐색합니다.
2.  **`src/train.py`**: `app.ipynb`에서 검증된 최상의 모델 구조를 기반으로 모델을 훈련시키고, `models/` 디렉토리에 `predictive_model.joblib` 파일로 저장합니다.
3.  **`src/predict.py`**: 저장된 모델을 불러와, 사용자가 입력하는 새로운 센서 값에 대한 고장 여부를 실시간으로 예측하고 결과를 반환합니다.

## 사용법

### 1. 환경 설정

프로젝트 환경을 설정하려면 다음 단계를 따르세요.

```bash
# 1. (필요시) 리포지토리 복제
# git clone <repository_url>
# cd Predictive-Maintenance-for-Industry

# 2. 가상 환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 3. 필요 라이브러리 설치
pip install -r requirements.txt
```

### 2. 모델 훈련

예측 애플리케이션을 실행하기 전, 반드시 모델 훈련을 먼저 수행해야 합니다.

```bash
python src/train.py
```

성공적으로 실행되면, `models/` 디렉토리에 `predictive_model.joblib`과 `model_columns.joblib` 파일이 생성됩니다.

### 3. 실시간 고장 예측

훈련된 모델을 사용하여 새로운 데이터에 대한 고장을 예측합니다. `src/predict.py`를 실행하고, 터미널의 안내에 따라 센서 값을 입력하세요.

```bash
python src/predict.py
```

**CLI 실행 예시:**
```
--- Machine Failure Prediction ---
Trained model and columns loaded successfully.

Please enter the following sensor values:
- Air temperature [K]: 301.5
- Process temperature [K]: 310.8
- Rotational speed [rpm]: 1400
- Torque [Nm]: 55.2
- Tool wear [min]: 180
- Machine Type (H, M, or L): L

--- Prediction Result ---
Status: [FAILURE PREDICTED]
Confidence: 82.50%

Recommendation: Investigate the machine for potential issues.
-------------------------
```

### 4. 데이터 분석 및 모델링 과정 확인

데이터 분석, 모델 선택, 하이퍼파라미터 튜닝 등 전체 모델 개발 과정이 궁금하다면 `notebooks/app.ipynb` 노트북을 참고하세요.

```bash
jupyter notebook notebooks/app.ipynb
```

## 모델 성능

- **모델**: `RandomForestClassifier` (불균형 데이터 처리를 위해 `class_weight='balanced'` 적용)
- **주요 성능 지표**: 실제 고장을 놓치지 않는 것이 중요하므로 **재현율(Recall)** 을 핵심 지표로 사용합니다.

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0 (정상)      | 0.99      | 0.99   | 0.99     | 1932    |
| **1 (고장)**  | **0.67**  | **0.82**   | **0.74** | 68      |
|---------------|-----------|--------|----------|---------|
| **Accuracy**  |           |        | **0.99** | 2000    |


## 향후 개선 방향
- **모델 교체 및 비교**: `XGBoost`, `LightGBM` 등 다른 앙상블 모델을 적용하여 성능을 비교하고, 가장 우수한 모델을 최종 선택합니다.
- **웹 애플리케이션 개발**: `Streamlit`이나 `Flask`를 사용하여 사용자가 더 쉽게 상호작용할 수 있는 웹 기반 UI를 개발합니다.
- **실시간 처리 아키텍처 구상**: `Kafka`와 같은 메시징 큐와 연동하여 실제 산업 현장처럼 스트리밍 데이터를 실시간으로 받아 예측하는 시스템을 설계합니다.

## 데이터 소스
- **Kaggle**: [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/fedesoriano/ai4i-2020-predictive-maintenance-dataset)