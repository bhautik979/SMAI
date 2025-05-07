import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_recall_curve
)
import os

def load_model(model_path="/content/best_model.pkl"):
    """
    Load the trained model from a pickle file
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def prepare_input_data(csv_path, drop_columns=None):
    """
    Load and prepare the input CSV file for prediction
    """
    try:
        # Load data
        data = pd.read_csv(csv_path)
        print(f"Loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
        
        # Convert date to datetime if it exists
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            if 'Year' not in data.columns:
                data['Year'] = data['Date'].dt.year
        
        # Handle missing values
        data.replace(['unknown', 'none', -1, ''], np.nan, inplace=True)
        
        # Fill numerical missing values with median
        num_cols = data.select_dtypes(include=np.number).columns
        for col in num_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        # Fill categorical missing values with 'missing'
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna('missing', inplace=True)
        
        # Create additional features if they were used in training
        data = create_features(data)
        
        # Separate target if it exists in the data
        target_values = None
        if 'Win?' in data.columns:
            target_values = data['Win?'].copy()
            if drop_columns is None:
                drop_columns = ['Win?']
            elif 'Win?' not in drop_columns:
                drop_columns.append('Win?')
        
        # Drop non-feature columns
        if drop_columns:
            data = data.drop(columns=[col for col in drop_columns if col in data.columns], errors='ignore')
        
        return data, target_values
        
    except Exception as e:
        print(f"Error preparing input data: {e}")
        return None, None

def create_features(df):
    """
    Create the same derived features that were used during training
    """
    # Only create features if the required base columns exist
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
        df['Runs_to_ERA_Ratio'] = df['Total Runs'] / (df['ERA'] + 0.1)  # Adding 0.1 to avoid division by zero
    
    if all(col in df.columns for col in ['AVG/5 Players', 'OBP/5 Players', 'SLG/5 Players']):
        df['Recent_Performance'] = df['AVG/5 Players'] * df['OBP/5 Players'] * df['SLG/5 Players']
    
    if all(col in df.columns for col in ['Opposing K/9', 'Opposing BB/9']):
        df['Pitching_Quality'] = df['Opposing K/9'] / (df['Opposing BB/9'] + 0.1)
    
    if all(col in df.columns for col in ['WAR', 'Opposing War']):
        df['Teams_WAR_Diff'] = abs(df['WAR'] - df['Opposing War'])
    
    return df

def evaluate_model(model, X, y_true):
    """
    Evaluate the model on test data and return performance metrics
    """
    if model is None or X is None or y_true is None:
        print("Cannot evaluate model: missing model, input data, or target values")
        return None
    
    results = {}
    
    # Make predictions
    y_pred = model.predict(X)
    results['predictions'] = y_pred
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X)
            results['probabilities'] = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
        except:
            results['probabilities'] = None
    else:
        results['probabilities'] = None
    
    # Calculate metrics
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['precision'] = precision_score(y_true, y_pred, zero_division=0)
    results['recall'] = recall_score(y_true, y_pred, zero_division=0)
    results['f1'] = f1_score(y_true, y_pred, zero_division=0)
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # ROC AUC if probabilities are available
    if results['probabilities'] is not None:
        results['roc_auc'] = roc_auc_score(y_true, results['probabilities'])
    else:
        results['roc_auc'] = None
    
    return results

def visualize_results(results, model_name="Loaded Model"):
    """
    Visualize the performance metrics
    """
    if results is None:
        return
    
    # Create directory for results
    os.makedirs("evaluation_results", exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('evaluation_results/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 2. ROC Curve (if probabilities are available)
    if results['probabilities'] is not None and results['roc_auc'] is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, results['probabilities'])
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {results["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('evaluation_results/roc_curve.png', dpi=300)
        plt.close()
    
    # 3. Performance Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] 
    values = [results.get(metric, 0) for metric in metrics]
    
    # Remove None values
    valid_metrics = []
    valid_values = []
    for m, v in zip(metrics, values):
        if v is not None:
            valid_metrics.append(m)
            valid_values.append(v)
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(valid_metrics)))
    bars = plt.bar(valid_metrics, valid_values, color=colors)
    plt.ylim([0, 1.05])
    plt.title(f'Performance Metrics - {model_name}')
    plt.ylabel('Score')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('evaluation_results/performance_metrics.png', dpi=300)
    plt.close()
    
def print_summary(results, model):
    """
    Print a summary of the model evaluation
    """
    if results is None:
        return
    
    print("\n" + "="*50)
    print("MODEL EVALUATION SUMMARY")
    print("="*50)
    
    # Print model type
    model_type = type(model).__name__
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        model_type = f"Pipeline with {type(model.named_steps['classifier']).__name__}"
    print(f"Model Type: {model_type}")
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    if results['roc_auc'] is not None:
        print(f"  ROC AUC:   {results['roc_auc']:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    # Print classification report
    if 'predictions' in results:
        print("\nClassification Report:")
        print(classification_report(y_true, results['predictions']))
    
    print("\nVisualization files saved in 'evaluation_results' directory")
    print("="*50)

if __name__ == "__main__":
    # 1. Load the saved model
    model = load_model()
    
    if model is None:
        print("Failed to load model. Exiting.")
        exit(1)
    
    # 2. Load and prepare the test data
    # Define columns to drop (non-feature columns)
    columns_to_drop = ['Date', 'Year', 'Offensive Team', 'Defensive Team', 'Runs Scored']
    
    # Load the input CSV file
    input_file = input("Enter path to your input CSV file: ")  # e.g., "test_data.csv"
    X_test, y_true = prepare_input_data(input_file, drop_columns=columns_to_drop)
    
    if X_test is None:
        print("Failed to prepare input data. Exiting.")
        exit(1)
    
    # Handle case where we have data but no labels (pure prediction)
    if y_true is None:
        print("\nNo target column ('Win?') found in input data. Performing prediction only.")
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
                win_probs = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba
                
                # Add predictions and probabilities to original data
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
        
        # Save predictions to CSV
        output_file = "prediction_results.csv"
        result_df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to {output_file}")
        
    else:
        # 3. Evaluate the model
        results = evaluate_model(model, X_test, y_true)
        
        # 4. Visualize results
        visualize_results(results)
        
        # 5. Print summary
        print_summary(results, model)
