import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

def train_model():
    """
    This function trains a predictive maintenance model and saves it.
    It performs the following steps:
    1. Loads the dataset from 'ai4i2020.csv'.
    2. Preprocesses the data:
        - Drops unnecessary columns ('UDI', 'Product ID').
        - Applies one-hot encoding to the 'Type' column.
    3. Splits the data into training and testing sets.
    4. Trains a RandomForestClassifier with balanced class weights.
    5. Evaluates the model on the test set and prints the performance.
    6. Saves the trained model and the feature columns to disk.
    """
    print("Starting model training process...")

    # 1. Load Data
    try:
        df = pd.read_csv('../data/ai4i2020.csv')
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: 'ai4i2020.csv' not found. Make sure it is in the 'data/' directory.")
        return

    # 2. Preprocess Data
    print("Preprocessing data...")
    # Drop unnecessary columns and apply one-hot encoding
    df_model = df.drop(['UDI', 'Product ID'], axis=1)
    df_model = pd.get_dummies(df_model, columns=['Type'], drop_first=True)

    # Separate features (X) and target (y)
    X = df_model.drop('Machine failure', axis=1)
    y = df_model['Machine failure']
    
    # Store column order for the prediction script
    model_columns = X.columns
    joblib.dump(model_columns, '../models/model_columns.joblib')
    print("Model columns saved to 'models/model_columns.joblib'.")

    # 3. Split Data
    # stratify=y ensures the same proportion of failures in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")

    # 4. Train Model
    print("Training RandomForestClassifier...")
    # Using class_weight='balanced' is crucial for imbalanced datasets
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training completed.")

    # 5. Evaluate Model
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%
")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("------------------------\n")

    # 6. Save Model
    joblib.dump(model, '../models/predictive_model.joblib')
    print("Trained model saved successfully as 'models/predictive_model.joblib'.")

if __name__ == '__main__':
    train_model()
