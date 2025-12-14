import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')
print("라이브러리 로드 완료")

# CSV 파일 불러오기
try:
    df = pd.read_csv('ai4i2020.csv')
except FileNotFoundError:
    print("오류: 'ai4i2020.csv' 파일을 'hankook-project' 폴더에 넣어주세요.")

# 데이터 앞부분 5줄 확인
print("데이터 샘플:")
print(df.head())

# 데이터 기본 정보 확인
print("\n데이터 정보:")
df.info()

# 고장(1)과 정상(0) 데이터 개수 확인
failure_counts = df['Machine failure'].value_counts()
print("고장 데이터 분포:\n", failure_counts)

# 고장 비율 계산
failure_rate = (failure_counts[1] / len(df)) * 100
print(f"\n전체 데이터 중 고장 비율: {failure_rate:.2f}%")

# 고장 분포 시각화
plt.figure(figsize=(8, 5))
sns.countplot(x='Machine failure', data=df)
plt.title('Machine Failure Distribution (0: No Failure, 1: Failure)')
plt.show()

# 분석할 주요 피처(센서 데이터) 목록
features_to_plot = ['Torque [Nm]', 'Rotational speed [rpm]', 'Tool wear [min]', 'Process temperature [K]']

# Box plot으로 시각화
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features_to_plot):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='Machine failure', y=feature, data=df)
    plt.title(f'{feature} vs. Machine Failure')
plt.show()

# 'Product ID'와 같은 불필요한 컬럼 제거 및 'Type' 컬럼을 숫자로 변환
df_model = df.drop(['UDI', 'Product ID'], axis=1)
df_model = pd.get_dummies(df_model, columns=['Type'], drop_first=True)

# 데이터를 피처(X)와 타겟(y)으로 분리
X = df_model.drop('Machine failure', axis=1)
y = df_model['Machine failure']

# 학습용 데이터와 테스트용 데이터로 8:2 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("데이터 준비 완료")
print(f"학습 데이터 개수: {len(X_train)}")
print(f"테스트 데이터 개수: {len(X_test)}")

# 랜덤 포레스트 분류 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 테스트 데이터로 예측 수행
y_pred = model.predict(X_test)

# 모델 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 예측 정확도: {accuracy * 100:.2f}%\n")
print("성능 상세 리포트:\n", classification_report(y_test, y_pred))

# 피처 중요도 계산
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

print("고장 예측에 중요한 변수 Top 5:")
print(feature_importances.head())

# 피처 중요도 시각화
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature Importances for Predicting Machine Failure')
plt.show()
