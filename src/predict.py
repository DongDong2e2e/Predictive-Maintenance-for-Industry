import joblib
import pandas as pd
import os

def get_validated_input(prompt: str, input_type: type = float):
    """
    Prompts the user for input and validates it against the specified type.
    Loops until a valid input is received.

    Args:
        prompt (str): The message to display to the user.
        input_type (type): The desired type of the input (e.g., float, str).

    Returns:
        The validated user input of the specified type.
    """
    while True:
        user_input = input(prompt)
        if input_type == float:
            try:
                return float(user_input)
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
        elif input_type == str:
            # For machine type, validate against specific allowed values
            allowed_values = ['H', 'M', 'L']
            if user_input.upper() in allowed_values:
                return user_input.upper()
            else:
                print(f"Invalid input. Please enter one of {', '.join(allowed_values)}.")
        else:
            return user_input

def predict_failure():
    """
    This function loads a pre-trained model and predicts machine failure based on user input.
    It now includes robust input validation.
    """
    print("--- Machine Failure Prediction ---")
    
    # Define paths relative to the script's directory for robustness
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..')) # Go up one level from src/
    models_dir = os.path.join(project_root, 'models')

    # 1. Load Model and Columns
    try:
        model_path = os.path.join(models_dir, 'predictive_model.joblib')
        columns_path = os.path.join(models_dir, 'model_columns.joblib')
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
        print("Trained model and columns loaded successfully.")
    except FileNotFoundError:
        print("\nError: Model files not found.")
        print("Please run 'python src/train.py' first to train and save the model.")
        return

    # 2. Get User Input with Validation
    print("\nPlease enter the following sensor values:")
    
    input_data = {
        'Air temperature [K]': [get_validated_input("- Air temperature [K]: ")],
        'Process temperature [K]': [get_validated_input("- Process temperature [K]: ")],
        'Rotational speed [rpm]': [get_validated_input("- Rotational speed [rpm]: ")],
        'Torque [Nm]': [get_validated_input("- Torque [Nm]: ")],
        'Tool wear [min]': [get_validated_input("- Tool wear [min]: ")]
    }
    
    type_input = get_validated_input("- Machine Type (H, M, or L): ", str)
    
    # Set the one-hot encoded columns based on user input
    input_data['Type_L'] = 1 if type_input == 'L' else 0
    input_data['Type_M'] = 1 if type_input == 'M' else 0

    # 3. Create DataFrame
    input_df = pd.DataFrame(input_data)

    # 4. Align DataFrame Columns
    query_df = pd.DataFrame(columns=model_columns)
    query_df = pd.concat([query_df, input_df], ignore_index=True, sort=False)
    query_df = query_df.fillna(0)
    query_df = query_df[model_columns]

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