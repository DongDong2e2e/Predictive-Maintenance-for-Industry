# 예측 유지보수 프로젝트

## 프로젝트 목표
이 프로젝트는 기계 장비의 센서 데이터를 분석하여 잠재적인 고장을 예측하는 머신러닝 모델을 개발하는 것을 목표로 합니다. 이를 통해 선제적인 유지보수 조치를 취함으로써 설비의 안정성을 높이고, 예기치 않은 가동 중단으로 인한 손실을 최소화하는 솔루션을 제시합니다.

## 프로젝트 동기
발전소와 같은 대규모 산업 설비에서 예기치 못한 고장은 막대한 경제적 손실과 안전 문제로 이어질 수 있습니다. 본 프로젝트는 데이터 기반의 예측 유지보수 기술이 어떻게 발전소의 안정적이고 효율적인 운영에 기여할 수 있는지 보여주기 위해 시작되었습니다. AI/ML 기술을 활용하여 설비의 상태를 실시간으로 진단하고, 고장 가능성을 사전에 예측하는 역량을 발전소의 실제 운영 환경에 적용하고자 하는 열망을 담고 있습니다.

## 파일:
- `ai4e2020.csv`: 모델 훈련 및 평가에 사용된 데이터셋입니다.
- `app.ipynb`: 탐색적 데이터 분석, 모델 개발 및 실험 과정을 담은 Jupyter Notebook입니다.
- `app.py`: 개발된 모델을 실제 데이터에 적용하고 예측 결과를 도출하는 Python 애플리케이션입니다.
- `requirements.txt`: 프로젝트 실행에 필요한 Python 라이브러리 목록입니다.

## 데이터 소스
본 프로젝트는 Kaggle에 공개된 **AI4I 2020 Predictive Maintenance Dataset**을 사용하였습니다.
- **데이터셋 링크:** [https://www.kaggle.com/datasets/fedesoriano/ai4i-2020-predictive-maintenance-dataset](https://www.kaggle.com/datasets/fedesoriano/ai4i-2020-predictive-maintenance-dataset)

이 데이터셋은 실제 산업 환경에서 수집된 데이터를 모사하여 제작되었으며, 다양한 센서값과 기계 고장 여부를 포함하고 있습니다.

## 설치

프로젝트 환경을 설정하려면 다음 단계를 따르세요:

1.  **리포지토리 복제 (해당하는 경우):**
    ```bash
    git clone <repository_url>
    cd Predictive Maintanance
    ```

2.  **가상 환경 생성:**
    ```bash
    python -m venv venv
    ```

3.  **가상 환경 활성화:**
    -   macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    -   Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **의존성 설치:**
    ```bash
    pip install -r requirements.txt
    ```

## 사용법

### Jupyter Notebook (`app.ipynb`)

데이터 분석 및 모델 개발 과정을 확인하려면:

1.  가상 환경을 활성화하세요.
2.  Jupyter Notebook을 실행하세요:
    ```bash
    jupyter notebook
    ```
3.  브라우저에서 `app.ipynb` 파일을 열어 코드를 실행하고 결과를 확인할 수 있습니다.

### Python 애플리케이션 (`app.py`)

개발된 모델을 사용하여 예측을 수행하려면:

1.  가상 환경을 활성화하세요.
2.  아래 명령어로 Python 스크립트를 실행하세요:
    ```bash
    python app.py
    ```
    (참고: `app.py`의 구체적인 동작 방식은 파일 내용을 참고해 주시기 바랍니다.)

## 향후 개선 방향
- (여기에 모델 성능 개선, 추가 기능 개발 등 향후 계획을 작성할 수 있습니다.)

## 라이선스
(라이선스 정보를 여기에 추가하세요)
