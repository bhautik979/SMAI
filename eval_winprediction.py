import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import os

def load_model(model_path="/content/best_model.pkl"):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def prepare_input_data(csv_path, drop_columns=None, nrows=5):
    try:
        data = pd.read_csv(csv_path, nrows=nrows)
        print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")

        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            if 'Year' not in data.columns:
                data['Year'] = data['Date'].dt.year

        data.replace(['unknown', 'none', -1, ''], np.nan, inplace=True)

        num_cols = data.select_dtypes(include=np.number).columns
        for col in num_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)

        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna('missing', inplace=True)

        data = create_features(data)

        target_values = None
        if 'Win?' in data.columns:
            target_values = data['Win?'].copy()
            if drop_columns is None:
                drop_columns = ['Win?']
            elif 'Win?' not in drop_columns:
                drop_columns.append('Win?')

        if drop_columns:
            data = data.drop(columns=[col for col in drop_columns if col in data.columns], errors='ignore')

        return data, target_values

    except Exception as e:
        print(f"Error preparing input data: {e}")
        return None, None

def create_features(df):
    if all(col in df.columns for col in ['AVG', 'AVG/Week']):
        df['Team_Diff_AVG'] = df['AVG'] - df['AVG/Week']
    if all(col in df.columns for col in ['OBP', 'OBP/Week']):
        df['Team_Diff_OBP'] = df['OBP'] - df['OBP/Week']
    if all(col in df.columns for col in ['SLG', 'SLG/Week']):
        df['Team_Diff_SLG'] = df['SLG'] - df['SLG/Week']
    if all(col in df.columns for col in ['WAR', 'WAR/Week']):
        df['Team_Diff_WAR'] = df['WAR'] - df['WAR/Week']
    if all(col in df.columns for col in ['WRC+', 'WRC+/Week']):
        df['Team_Diff_WRC'] = df['WRC+'] - df['WRC+/Week']
    if all(col in df.columns for col in ['Total Runs', 'ERA']):
        df['Runs_to_ERA_Ratio'] = df['Total Runs'] / (df['ERA'] + 0.1)
    if all(col in df.columns for col in ['AVG/5 Players', 'OBP/5 Players', 'SLG/5 Players']):
        df['Recent_Performance'] = df['AVG/5 Players'] * df['OBP/5 Players'] * df['SLG/5 Players']
    if all(col in df.columns for col in ['Opposing K/9', 'Opposing BB/9']):
        df['Pitching_Quality'] = df['Opposing K/9'] / (df['Opposing BB/9'] + 0.1)
    if all(col in df.columns for col in ['WAR', 'Opposing War']):
        df['Teams_WAR_Diff'] = abs(df['WAR'] - df['Opposing War'])
    return df

if __name__ == "__main__":
    model = load_model()
    if model is None:
        exit(1)

    columns_to_drop = ['Date', 'Year', 'Offensive Team', 'Defensive Team', 'Runs Scored']
    input_file = input("Enter path to your input CSV file: ")

    X_test, y_true = prepare_input_data(input_file, drop_columns=columns_to_drop, nrows=5)
    if X_test is None:
        exit(1)

    y_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)
            win_probs = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
            result_df = pd.DataFrame({
                'Predicted_Outcome': ['Win' if p == 1 else 'Loss' for p in y_pred],
                'Win_Probability': win_probs
            })
        except:
            result_df = pd.DataFrame({
                'Predicted_Outcome': ['Win' if p == 1 else 'Loss' for p in y_pred]
            })
    else:
        result_df = pd.DataFrame({
            'Predicted_Outcome': ['Win' if p == 1 else 'Loss' for p in y_pred]
        })

    output_file = "prediction_results.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\nPredictions for first 5 rows saved to {output_file}")
