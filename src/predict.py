import joblib
import pandas as pd

def predict_failure():
    """
    This function loads a pre-trained model and predicts machine failure based on user input.
    Steps:
    1. Loads the trained RandomForest model and the model columns.
    2. Prompts the user to enter sensor values for prediction.
    3. Creates a DataFrame from the user input.
    4. Ensures the input DataFrame has the correct columns and order.
    5. Predicts the failure status and the probability.
    6. Displays the prediction to the user.
    """
    print("--- Machine Failure Prediction ---")
    
    # 1. Load Model and Columns
    try:
        model = joblib.load('../models/predictive_model.joblib')
        model_columns = joblib.load('../models/model_columns.joblib')
        print("Trained model and columns loaded successfully.")
    except FileNotFoundError:
        print("\nError: Model files not found.")
        print("Please run 'python src/train.py' first to train and save the model.")
        return

    # 2. Get User Input
    print("\nPlease enter the following sensor values:")
    
    input_data = {
        'Air temperature [K]': [float(input("- Air temperature [K]: "))],
        'Process temperature [K]': [float(input("- Process temperature [K]: "))],
        'Rotational speed [rpm]': [float(input("- Rotational speed [rpm]: "))],
        'Torque [Nm]': [float(input("- Torque [Nm]: "))],
        'Tool wear [min]': [float(input("- Tool wear [min]: "))]
    }
    
    # For the 'Type' feature, which was one-hot encoded
    type_input = input("- Machine Type (H, M, or L): ").upper()
    
    # Set the one-hot encoded columns based on user input
    input_data['Type_L'] = [1 if type_input == 'L' else 0]
    input_data['Type_M'] = [1 if type_input == 'M' else 0]

    # 3. Create DataFrame
    input_df = pd.DataFrame(input_data)

    # 4. Align DataFrame Columns
    # Ensure the input_df has the same columns in the same order as the training data
    query_df = pd.DataFrame(columns=model_columns)
    query_df = pd.concat([query_df, input_df], ignore_index=True, sort=False)
    query_df = query_df.fillna(0) # Fill any missing columns with 0
    query_df = query_df[model_columns] # Enforce column order

    # 5. Make Prediction
    prediction = model.predict(query_df)
    probability = model.predict_proba(query_df)

    # 6. Display Result
    print("\n--- Prediction Result ---")
    if prediction[0] == 1:
        print("Status: [FAILURE PREDICTED]")
        print(f"Confidence: {probability[0][1] * 100:.2f}%")
        print("\nRecommendation: Investigate the machine for potential issues.")
    else:
        print("Status: [NORMAL OPERATION]")
        print(f"Confidence: {probability[0][0] * 100:.2f}%")
        print("\nRecommendation: No immediate action required.")
    print("-------------------------\n")


if __name__ == '__main__':
    predict_failure()
