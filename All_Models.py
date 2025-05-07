import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, confusion_matrix,
                           classification_report, roc_auc_score,
                           roc_curve, auc, precision_recall_curve,
                           log_loss)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
import warnings
from sklearn.linear_model import LogisticRegression, RidgeClassifier # Added RidgeClassifier for completeness if used
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, BaggingClassifier,
    AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.base import clone # Import clone
warnings.filterwarnings('ignore')

from sklearn.metrics import (precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix)
import numpy as np

import pickle
import os
import joblib

# Data Loading and Cleaning for CSV
def load_and_clean_data(file_path='/content/stats.csv'):
    try:
        # Read CSV file
        data = pd.read_csv("/content/stats.csv")
        print(f"Successfully loaded {len(data)} records from {file_path}")

        # Convert date to datetime
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
        data['Year'] = data['Date'].dt.year

        # The target variable is already in the dataset as 'Win?'
        # 1 = win, 0 = loss

        # Handle missing values
        data.replace(['unknown', 'none', -1, ''], np.nan, inplace=True)

        # Fill numerical missing values
        num_cols = data.select_dtypes(include=np.number).columns
        for col in num_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)

        # Fill categorical missing values
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna('missing', inplace=True)

        # Print dataset info
        print("\nDataset Summary:")
        print(f"Shape: {data.shape}")
        print(f"Columns: {', '.join(data.columns)}")
        print(f"Number of games: {len(data)}")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

        return data

    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Create additional features based on the dataset
def create_features(df):
    # Create some team performance metrics
    df['Team_Diff_AVG'] = df['AVG'] - df['AVG/Week']
    df['Team_Diff_OBP'] = df['OBP'] - df['OBP/Week']
    df['Team_Diff_SLG'] = df['SLG'] - df['SLG/Week']
    df['Team_Diff_WAR'] = df['WAR'] - df['WAR/Week']
    df['Team_Diff_WRC'] = df['WRC+'] - df['WRC+/Week']

    # Create matchup features
    df['Runs_to_ERA_Ratio'] = df['Total Runs'] / (df['ERA'] + 0.1)  # Adding 0.1 to avoid division by zero

    # Create trend features
    df['Recent_Performance'] = df['AVG/5 Players'] * df['OBP/5 Players'] * df['SLG/5 Players']
    df['Pitching_Quality'] = df['Opposing K/9'] / (df['Opposing BB/9'] + 0.1)

    # Feature to capture if teams are evenly matched
    df['Teams_WAR_Diff'] = abs(df['WAR'] - df['Opposing War'])

    return df

# Custom year-based data splitting with fallback options
def split_data(data, train_years=(2000, 2020), val_years=(2021, 2022), test_years=(2022, 2024)):
    if 'Year' not in data.columns:
        print("Adding Year column from Date...")
        data['Year'] = data['Date'].dt.year

    available_years = sorted(data['Year'].unique())
    if not available_years: # Handle case where data is empty or 'Year' column could not be populated
        print("WARNING: No year data available. Falling back to random split.")
        train_val, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data[target] if target in data else None)
        train_data, val_data = train_test_split(train_val, test_size=0.2, random_state=42, stratify=train_val[target] if target in train_val else None)
        print(f"Using random split: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}")
        return train_data, val_data, test_data

    min_year_data = min(available_years)
    max_year_data = max(available_years)
    print(f"\nDataset contains years: {available_years}")
    print(f"Year range in data: {min_year_data} to {max_year_data}")

    train = data[data['Year'].between(train_years[0], train_years[1])]
    val = data[data['Year'].between(val_years[0], val_years[1])]
    test = data[data['Year'].between(test_years[0], test_years[1])]

    empty_splits = []
    if len(train) == 0: empty_splits.append(f"Train ({train_years[0]}-{train_years[1]})")
    if len(val) == 0: empty_splits.append(f"Validation ({val_years[0]}-{val_years[1]})")
    if len(test) == 0: empty_splits.append(f"Test ({test_years[0]}-{test_years[1]})")

    if empty_splits:
        print(f"\nWARNING: The following year-based splits are empty: {', '.join(empty_splits)}")
        print("Falling back to chronological split (70% train, 15% validation, 15% test).")

        data_sorted = data.sort_values('Date') # Sort by Date for chronological split
        total_rows = len(data_sorted)
        train_size = int(0.7 * total_rows)
        val_size = int(0.15 * total_rows)

        train = data_sorted.iloc[:train_size]
        val = data_sorted.iloc[train_size : train_size + val_size]
        test = data_sorted.iloc[train_size + val_size:]
    else:
        print(f"\nUsing year-based split:")
        print(f"Train: {train_years[0]}-{train_years[1]} ({len(train)} samples)")
        print(f"Validation: {val_years[0]}-{val_years[1]} ({len(val)} samples)")
        print(f"Test: {test_years[0]}-{test_years[1]} ({len(test)} samples)")

    print(f"\nFinal split sizes:")
    print(f"Train: {len(train)} games ({len(train)/len(data):.1%} of data)")
    print(f"Validation: {len(val)} games ({len(val)/len(data):.1%} of data)")
    print(f"Test: {len(test)} games ({len(test)/len(data):.1%} of data)")

    min_required = 10
    for split_name, split_df in [("Train", train), ("Validation", val), ("Test", test)]:
        if len(split_df) < min_required:
            print(f"WARNING: {split_name} split has only {len(split_df)} samples, which may be too few for robust evaluation or training.")
        if len(split_df) == 0 and split_name == "Train": # Critical if training set is empty
             raise ValueError("Training data split is empty. Cannot proceed.")

    return train, val, test

# Feature Selection
def select_features(data):
    target = 'Win?'
    features_to_drop = ['Date', 'Year', 'Offensive Team', 'Defensive Team', 'Runs Scored']

    # Ensure target is not in features_to_drop if it exists
    if target in features_to_drop:
        features_to_drop.remove(target)

    categorical_features = [] # Assuming no explicit categorical features for now based on original code
                               # If there were, they would be identified like:
                               # data.select_dtypes(include=['object', 'category']).columns.tolist()
                               # and removed from numerical_features list.

    numerical_features = [col for col in data.columns
                         if col not in categorical_features + features_to_drop + [target]
                         and pd.api.types.is_numeric_dtype(data[col])]

    print(f"\n{len(numerical_features)} numerical features selected: {numerical_features}")
    print(f"{len(categorical_features)} categorical features selected: {categorical_features}")

    return numerical_features, categorical_features, target, features_to_drop

# Add this function after the select_features function
def plot_feature_correlation_heatmap(data, target, top_n=20):
    """
    Creates a correlation heatmap for the top N features most correlated with the target variable.
    
    Args:
        data: DataFrame containing the features and target
        target: Name of the target variable
        top_n: Number of top features to include (default: 20)
    
    Returns:
        The matplotlib figure
    """
    # Calculate correlation matrix for numeric features
    corr_matrix = data.corr(numeric_only=True)
    
    # Get correlation with target variable
    target_corr = corr_matrix[target].drop(target)
    
    # Get top N features most correlated with target (absolute value)
    top_features = target_corr.abs().sort_values(ascending=False).head(top_n).index.tolist()
    
    # Always include the target variable at the end
    if target in top_features:
        top_features.remove(target)
    top_features.append(target)
    
    # Select the correlation matrix for top features
    top_corr = corr_matrix.loc[top_features, top_features]
    
    # Create the heatmap with improved visibility
    plt.figure(figsize=(16, 14))
    
    # Create a custom diverging colormap with distinct colors
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Plot the heatmap with enhanced settings
    heatmap = sns.heatmap(
        top_corr, 
        annot=True,
        cmap=cmap,
        fmt=".2f",
        linewidths=0.8,
        square=True,
        vmin=-1, 
        vmax=1,
        center=0,
        annot_kws={"size": 9}
    )
    
    # Adjust the plot
    plt.title(f'Top {top_n} Features Correlation Heatmap with {target}', fontsize=18, pad=20)
    
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(rotation=90, fontsize=10)
    
    # Add a colorbar legend with clearer label
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Correlation Coefficient', fontsize=12, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('top_features_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    return plt.gcf()

# Plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()

# Plot ROC curve
def plot_roc_curve(y_true, y_scores, model_name): # y_scores can be probabilities or decision_function outputs
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt.gcf()

# Plot accuracy vs loss chart
def plot_accuracy_vs_loss(model_name, train_metrics, val_metrics, test_metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    datasets = ['CV Train', 'Validation', 'Test'] # Changed 'Train' to 'CV Train'
    accuracies = [train_metrics['accuracy'], val_metrics['accuracy'], test_metrics['accuracy']]
    losses = [train_metrics['loss'], val_metrics['loss'], test_metrics['loss']]

    colors = ['#1f77b4', '#2ca02c', '#d62728'] # Adjusted colors

    ax1.bar(datasets, accuracies, color=colors)
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_ylim([0, 1.05])
    for i, v in enumerate(accuracies):
        if not np.isnan(v): ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    ax2.bar(datasets, losses, color=colors)
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{model_name} - Loss')
    min_loss = min(l for l in losses if not np.isnan(l)) if any(not np.isnan(l) for l in losses) else 0
    max_loss = max(l for l in losses if not np.isnan(l)) if any(not np.isnan(l) for l in losses) else 1
    ax2.set_ylim([min_loss * 0.9 if min_loss > 0 else -0.05, max_loss * 1.1 + 0.05])

    for i, v in enumerate(losses):
         if not np.isnan(v): ax2.text(i, v + (max_loss*0.02), f'{v:.3f}', ha='center', va='bottom')

    plt.suptitle(f'{model_name} Performance Metrics', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
    return fig

# Prepare X and y datasets
def prepare_X_y(df, target, features_to_drop):
    # Make a copy to avoid SettingWithCopyWarning if df is a slice
    df_copy = df.copy()

    # Identify columns to actually drop
    cols_to_drop_present = [col for col in features_to_drop + [target] if col in df_copy.columns]

    X = df_copy.drop(columns=cols_to_drop_present, errors='ignore')
    y = df_copy[target]
    return X, y

# Modify the calculate_metrics function to include precision, recall, and F1
def calculate_metrics(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    # Add precision, recall, and F1 calculations
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    loss = np.nan
    auc_val = np.nan
    y_scores_for_output = y_pred # Default if no probabilities/scores

    if hasattr(model, "predict_proba"):
        try:
            y_pred_proba = model.predict_proba(X)
            # For binary classification, probabilities for the positive class (class 1)
            y_scores = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2 else y_pred_proba

            loss = log_loss(y, y_scores)
            auc_val = roc_auc_score(y, y_scores)
            y_scores_for_output = y_scores
        except Exception as e:
            # print(f"Info: Could not calculate log_loss/AUC using predict_proba for a model. {e}")
            loss = 1 - accuracy  # Fallback loss
            auc_val = 0.5        # Fallback AUC
    elif hasattr(model, "decision_function"):
        try:
            y_decision_scores = model.decision_function(X)
            auc_val = roc_auc_score(y, y_decision_scores)
            y_scores_for_output = y_decision_scores
            loss = 1 - accuracy # log_loss typically requires probabilities
        except Exception as e:
            # print(f"Info: Could not use decision_function for AUC for a model. {e}")
            loss = 1 - accuracy
            auc_val = 0.5
    else: # No predict_proba or decision_function
        loss = 1 - accuracy
        auc_val = 0.5 # Cannot calculate ROC AUC properly without scores/probabilities
        # y_scores_for_output remains y_pred; ROC will be very basic
        try:
            # roc_auc_score can sometimes work with y_pred for a basic measure
            auc_val = roc_auc_score(y, y_pred)
        except:
            pass # auc_val remains 0.5

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': loss,
        'auc': auc_val,
        'predictions': y_pred,
        'probabilities': y_scores_for_output
    }

# Modify the train_and_evaluate_model function to track confusion matrices across folds
def train_and_evaluate_model(model_template, model_name, X_train_full, y_train_full, X_val, y_val, X_test, y_test, preprocessor, n_splits=10):
    try:
        print(f"\nTraining {model_name} with {n_splits}-fold Cross-Validation...")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_accuracies, cv_losses, cv_aucs = [], [], []
        cv_precisions, cv_recalls, cv_f1s = [], [], []
        cv_confusion_matrices = [] # Store confusion matrices from each fold

        for fold, (train_idx, cv_val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
            # print(f"  Fold {fold+1}/{n_splits}")
            X_cv_train, X_cv_val = X_train_full.iloc[train_idx], X_train_full.iloc[cv_val_idx]
            y_cv_train, y_cv_val = y_train_full.iloc[train_idx], y_train_full.iloc[cv_val_idx]

            # Clone preprocessor and model for each fold to ensure independence
            current_preprocessor = clone(preprocessor)
            current_model = clone(model_template)

            fold_pipeline = Pipeline([
                ('preprocessor', current_preprocessor),
                ('classifier', current_model)
            ])

            fold_pipeline.fit(X_cv_train, y_cv_train)

            fold_metrics = calculate_metrics(fold_pipeline, X_cv_val, y_cv_val)
            cv_accuracies.append(fold_metrics['accuracy'])
            cv_losses.append(fold_metrics['loss'])
            cv_aucs.append(fold_metrics['auc'])
            cv_precisions.append(fold_metrics['precision'])
            cv_recalls.append(fold_metrics['recall'])
            cv_f1s.append(fold_metrics['f1'])
            
            # Calculate and store confusion matrix for this fold
            fold_cm = confusion_matrix(y_cv_val, fold_metrics['predictions'])
            cv_confusion_matrices.append(fold_cm)

        # Aggregate CV results
        # Handle potential NaNs in metrics if some folds had issues
        avg_cv_accuracy = np.nanmean(cv_accuracies)
        avg_cv_loss = np.nanmean(cv_losses)
        avg_cv_auc = np.nanmean(cv_aucs)
        avg_cv_precision = np.nanmean(cv_precisions)
        avg_cv_recall = np.nanmean(cv_recalls)
        avg_cv_f1 = np.nanmean(cv_f1s)
        
        # Calculate average confusion matrix across all folds
        avg_cv_cm = np.mean(cv_confusion_matrices, axis=0).round().astype(int)

        cv_train_metrics = {
            'accuracy': avg_cv_accuracy,
            'precision': avg_cv_precision,
            'recall': avg_cv_recall,
            'f1': avg_cv_f1,
            'loss': avg_cv_loss,
            'auc': avg_cv_auc,
            'predictions': None, # Not applicable for overall CV
            'probabilities': None, # Not applicable for overall CV
            'confusion_matrix': avg_cv_cm # Store the average confusion matrix
        }

        print(f"Retraining {model_name} on the full training set ({len(X_train_full)} samples)...")
        # Final model: Use original preprocessor (it will be fit here) and a new clone of the model template
        final_preprocessor = clone(preprocessor) # Or use the one passed if it's not stateful before fit
        final_model_instance = clone(model_template)

        final_pipeline = Pipeline([
            ('preprocessor', final_preprocessor),
            ('classifier', final_model_instance)
        ])
        final_pipeline.fit(X_train_full, y_train_full)

        val_metrics = calculate_metrics(final_pipeline, X_val, y_val)
        test_metrics = calculate_metrics(final_pipeline, X_test, y_test)
        
        # Add confusion matrices to validation and test metrics
        val_metrics['confusion_matrix'] = confusion_matrix(y_val, val_metrics['predictions'])
        test_metrics['confusion_matrix'] = confusion_matrix(y_test, test_metrics['predictions'])

        print(f"\n=== {model_name} Results ===")
        print(f"Avg CV Accuracy: {cv_train_metrics['accuracy']:.3f}, Precision: {cv_train_metrics['precision']:.3f}, Recall: {cv_train_metrics['recall']:.3f}, F1: {cv_train_metrics['f1']:.3f}")
        print(f"Avg CV Loss: {cv_train_metrics['loss']:.3f}, Avg CV AUC: {cv_train_metrics['auc']:.3f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.3f}, Precision: {val_metrics['precision']:.3f}, Recall: {val_metrics['recall']:.3f}, F1: {val_metrics['f1']:.3f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.3f}, Precision: {test_metrics['precision']:.3f}, Recall: {test_metrics['recall']:.3f}, F1: {test_metrics['f1']:.3f}")

        print("\nAverage Cross-Validation Confusion Matrix:")
        print(cv_train_metrics['confusion_matrix'])
        
        print("\nTest Confusion Matrix:")
        print(test_metrics['confusion_matrix'])
        
        # Classification report for test set
        cr = classification_report(y_test, test_metrics['predictions'])
        print("\nTest Classification Report:")
        print(cr)

        # Create confusion matrix plots for CV average and test
        cv_cm_fig = plot_confusion_matrix(cv_train_metrics['confusion_matrix'], f"{model_name} - Avg CV")
        test_cm_fig = plot_confusion_matrix(test_metrics['confusion_matrix'], model_name)

        roc_fig = None
        if not np.all(np.isnan(test_metrics['probabilities'])): # Check if probabilities/scores are not all NaN
             # And ensure they are 1D array of scores, not original predictions if all fails
            if len(test_metrics['probabilities']) > 0 and isinstance(test_metrics['probabilities'][0], (np.number, float, int)):
                roc_fig = plot_roc_curve(y_test, test_metrics['probabilities'], model_name)

        # Modified to include precision, recall and F1 in plot
        # acc_loss_fig = plot_metrics_comparison(model_name, cv_train_metrics, val_metrics, test_metrics)

        return {
            'model': final_pipeline,
            'train_metrics': cv_train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'confusion_matrix': test_metrics['confusion_matrix'],
            'avg_cv_confusion_matrix': cv_train_metrics['confusion_matrix'],
            'cv_cm_fig': cv_cm_fig,
            'test_cm_fig': test_cm_fig,
            'roc_fig': roc_fig,
            # 'acc_loss_fig': acc_loss_fig,
            'cv_scores': {
                'accuracy': cv_accuracies, 
                'precision': cv_precisions,
                'recall': cv_recalls,
                'f1': cv_f1s,
                'loss': cv_losses, 
                'auc': cv_aucs
            }
        }

    except Exception as e:
        print(f"Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# Main execution
if __name__ == "__main__":
    data = load_and_clean_data('stats.csv')
    data = create_features(data)

    # Define target before splitting for stratified split fallback
    target_col_name = 'Win?'

    train_df, val_df, test_df = split_data(data)

    if train_df.empty:
        print("Critical Error: Training dataset is empty after split. Exiting.")
        exit()

    numerical_features, categorical_features, target_col_name, features_to_drop = select_features(train_df) # Use train_df to select features

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    if categorical_features: # If you identify categorical features later
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ], remainder='passthrough') # Passthrough other columns if any (should not be if features selected carefully)
    else:
        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numerical_features)],
            remainder='passthrough') # Ensure only specified numerical features are transformed

    X_train, y_train = prepare_X_y(train_df, target_col_name, features_to_drop)
    X_val, y_val = prepare_X_y(val_df, target_col_name, features_to_drop)
    X_test, y_test = prepare_X_y(test_df, target_col_name, features_to_drop)
    all_selected_features = numerical_features + categorical_features
    X_train = X_train[all_selected_features]
    X_val = X_val[all_selected_features]
    X_test = X_test[all_selected_features]

    print("\nGenerating correlation heatmap for top features...")
    corr_heatmap_fig = plot_feature_correlation_heatmap(data, target_col_name)
    plt.show()

    print(f"\nClass distribution in training data (y_train):")
    print(y_train.value_counts(normalize=True))

    models_templates = {
        "Decision Tree": DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42),
        "Decision Tree (Deep)": DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=5, class_weight='balanced', random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=0.1),
        "Logistic Regression (L1)": LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, class_weight='balanced', random_state=42, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, class_weight='balanced', min_samples_leaf=2),
        "Random Forest (Large)": RandomForestClassifier(n_estimators=100, max_depth=8, max_features='sqrt', random_state=42, class_weight='balanced', bootstrap=True),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42, subsample=0.8),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss'), # Removed use_label_encoder
        "XGBoost (Tuned)": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, gamma=1, reg_alpha=0.1, reg_lambda=1, random_state=42, eval_metric='logloss'), # Removed use_label_encoder
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, random_state=42, early_stopping=True, n_iter_no_change=10), # Added early stopping
        "Naive Bayes": GaussianNB(),
        # "SVC": SVC(probability=True, random_state=42, class_weight='balanced') # Example if you want to test SVM
    }

    results = {}
    for name, model_template in models_templates.items():
        # Check if X_train is empty before proceeding
        if X_train.empty or y_train.empty:
            print(f"Skipping {name} as training data is empty.")
            continue
        if X_val.empty or y_val.empty:
            print(f"Warning: Validation set is empty for {name}.")
        if X_test.empty or y_test.empty:
            print(f"Warning: Test set is empty for {name}.")


        result = train_and_evaluate_model(
            model_template, name, X_train, y_train, X_val, y_val, X_test, y_test, preprocessor)
        if result is not None:
            results[name] = result
            if result.get('cm_fig'): plt.close(result['cm_fig'])
            if result.get('roc_fig'): plt.close(result['roc_fig'])
            # if result.get('acc_loss_fig'): plt.close(result['acc_loss_fig'])



    if results:
        n_models = len(results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        # Combined Confusion Matrices Plot
        fig_cm_all, axes_cm_all = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)
        axes_cm_all = axes_cm_all.flatten() # Flatten to 1D array for easy indexing
        for i, (name, result) in enumerate(results.items()):
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes_cm_all[i])
            test_acc = result["test_metrics"]["accuracy"]
            test_auc = result["test_metrics"]["auc"]
            axes_cm_all[i].set_title(f'{name}\nAcc: {test_acc:.3f}, AUC: {test_auc:.3f}')
            axes_cm_all[i].set_ylabel('True Label')
            axes_cm_all[i].set_xlabel('Predicted Label')
        # Hide any unused subplots
        for j in range(i + 1, len(axes_cm_all)):
            fig_cm_all.delaxes(axes_cm_all[j])
        plt.tight_layout()
        plt.savefig('all_confusion_matrices.png', dpi=300)
        plt.show()

        # Accuracy Comparison Plot
        model_names = list(results.keys())
        cv_train_acc = [results[name]['train_metrics']['accuracy'] for name in model_names]
        val_acc = [results[name]['val_metrics']['accuracy'] for name in model_names]
        test_acc = [results[name]['test_metrics']['accuracy'] for name in model_names]

        x_indices = np.arange(len(model_names))
        width = 0.25

        plt.figure(figsize=(15, 8))
        plt.bar(x_indices - width, cv_train_acc, width, label='Avg CV Train Acc', color='skyblue')
        plt.bar(x_indices, val_acc, width, label='Validation Acc', color='lightgreen')
        plt.bar(x_indices + width, test_acc, width, label='Test Acc', color='salmon')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Accuracy Comparison (CV Train, Validation, Test)', fontsize=14)
        plt.xticks(x_indices, model_names, rotation=45, ha='right', fontsize=10)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('model_accuracy_comparison.png', dpi=300)
        plt.show()

        # Combined ROC Curves Plot
        plt.figure(figsize=(12, 10))
        for name, result in results.items():
            if result['roc_fig'] is not None and not np.all(np.isnan(result['test_metrics']['probabilities'])):
                y_true_test = y_test # Ensure y_test is available here
                y_scores_test = result['test_metrics']['probabilities']

                # Check if y_scores_test is valid for roc_curve
                if y_scores_test is not None and len(y_scores_test) == len(y_true_test) and \
                   not (len(np.unique(y_scores_test)) == 1 and np.unique(y_scores_test)[0] in [0,1]): # Avoid plotting if only hard predictions and all same
                    fpr, tpr, _ = roc_curve(y_true_test, y_scores_test)
                    roc_auc_val = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, label=f'{name} (Test AUC = {roc_auc_val:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for All Models (Test Set)', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.savefig('all_roc_curves.png', dpi=300)
        plt.show()

        # Find best model based on validation AUC (handle NaNs)
        valid_results = {k: v for k, v in results.items() if not np.isnan(v['val_metrics']['auc'])}
        if valid_results:
            best_model_name = max(valid_results, key=lambda x: valid_results[x]['val_metrics']['auc'])
            best_model_result = results[best_model_name]
            best_model_pipeline = best_model_result['model']
            print(f"\nBest model (based on Validation AUC): {best_model_name}")
            print(f"  Validation AUC: {best_model_result['val_metrics']['auc']:.3f}")
            print(f"  Test AUC: {best_model_result['test_metrics']['auc']:.3f}")
            print(f"  Test Accuracy: {best_model_result['test_metrics']['accuracy']:.3f}")

            # Create directory for saved models
            os.makedirs("saved_models", exist_ok=True)

            # Save the model to a file
            model_filename = f"saved_models/best_model.pkl"
            with open(model_filename, 'wb') as file:
                pickle.dump(best_model_pipeline, file)

            print(f"\nBest model saved successfully to: {model_filename}")


            if not X_test.empty:
                sample_idx = 0
                sample_X = X_test.iloc[[sample_idx]]
                sample_pred = best_model_pipeline.predict(sample_X)[0]

                print(f"\nExample Prediction for Game {test_df.index[sample_idx] if hasattr(test_df, 'index') else sample_idx}:")
                # Access original 'Offensive Team', 'Defensive Team' if available in test_df
                original_sample_info = test_df.iloc[[sample_idx]]
                if 'Offensive Team' in original_sample_info.columns and 'Defensive Team' in original_sample_info.columns:
                     print(f"  Teams: {original_sample_info['Offensive Team'].values[0]} vs {original_sample_info['Defensive Team'].values[0]}")

                print(f"  Predicted outcome: {'Win' if sample_pred else 'Loss'}")
                if hasattr(best_model_pipeline, "predict_proba"):
                    sample_proba = best_model_pipeline.predict_proba(sample_X)[0]
                    print(f"  Probability Win: {sample_proba[1]:.2%}")
                    print(f"  Probability Loss: {sample_proba[0]:.2%}")
                print(f"  Actual outcome: {'Win' if y_test.iloc[sample_idx] else 'Loss'}")


            # Feature importance for the best model
            classifier_in_pipeline = best_model_pipeline.named_steps['classifier']
            preprocessor_in_pipeline = best_model_pipeline.named_steps['preprocessor']

            if hasattr(classifier_in_pipeline, 'feature_importances_') or \
               (hasattr(classifier_in_pipeline, 'coef_') and classifier_in_pipeline.coef_.ndim == 1) or \
               (hasattr(classifier_in_pipeline, 'coef_') and classifier_in_pipeline.coef_.ndim == 2 and classifier_in_pipeline.coef_.shape[0] == 1) :

                if hasattr(classifier_in_pipeline, 'feature_importances_'):
                    importances = classifier_in_pipeline.feature_importances_
                else: # Linear models (coef_)
                    importances = classifier_in_pipeline.coef_[0] if classifier_in_pipeline.coef_.ndim == 2 else classifier_in_pipeline.coef_
                    importances = np.abs(importances) # Use absolute values for magnitude

                # Get feature names from preprocessor
                try:
                    num_feature_names = preprocessor_in_pipeline.named_transformers_['num'].get_feature_names_out(numerical_features)

                    feature_names_ordered = list(num_feature_names)

                    if categorical_features: # If OHE was used
                        cat_pipeline = preprocessor_in_pipeline.named_transformers_['cat']
                        ohe_step_name = [name for name, _ in cat_pipeline.steps if isinstance(_, OneHotEncoder)][0]
                        cat_feature_names = cat_pipeline.named_steps[ohe_step_name].get_feature_names_out(categorical_features)
                        feature_names_ordered.extend(list(cat_feature_names))

                    # Handle 'passthrough' columns if any were explicitly named and came through
                    # This part needs careful matching if remainder='passthrough' actually passed columns
                    # For now, assume explicit features match transformed features

                except Exception as e:
                    print(f"Could not get feature names from preprocessor: {e}. Using generic names.")
                    feature_names_ordered = [f"feature_{i}" for i in range(len(importances))]


                if len(feature_names_ordered) == len(importances):
                    feat_imp_df = pd.DataFrame({'feature': feature_names_ordered, 'importance': importances})
                    feat_imp_df = feat_imp_df.sort_values('importance', ascending=False).head(20)

                    plt.figure(figsize=(12, 8))
                    ax = sns.barplot(x='importance', y='feature', data=feat_imp_df, palette="viridis")
                    plt.title(f'Top 20 Feature Importances for {best_model_name}', fontsize=14)
                    plt.xlabel('Importance Score', fontsize=12)
                    plt.ylabel('Features', fontsize=12)
                    plt.grid(axis='x', linestyle='--', alpha=0.6)

                    # Add values on bars
                    for p in ax.patches:
                        width = p.get_width()
                        plt.text(width + 0.001,  # x-position (slightly after bar end)
                                 p.get_y() + p.get_height() / 2,  # y-position (middle of bar)
                                 f'{width:.3f}',  # formatted text
                                 va='center')

                    plt.tight_layout()
                    plt.savefig('enhanced_feature_importance.png', dpi=300)
                    plt.show()

                    print("\nTop 10 Feature Importance Scores:")
                    for _, row in feat_imp_df.head(10).iterrows():
                        print(f"  {row['feature']}: {row['importance']:.4f}")
                else:
                    print(f"Mismatch between number of feature names ({len(feature_names_ordered)}) and importances ({len(importances)}). Skipping importance plot.")
            else:
                print(f"The best model ({best_model_name}) does not have 'feature_importances_' or 'coef_' attribute.")
        else:
            print("\nCould not determine the best model as no models had valid validation AUCs.")


        def predict_game_outcome(pipeline_model, features_dict, all_feature_names_in_order):
            # Create a DataFrame with the features in the correct order and all columns
            game_df_input = pd.DataFrame(columns=all_feature_names_in_order)
            game_df_input.loc[0] = np.nan # Initialize with NaNs
            for key, value in features_dict.items():
                if key in game_df_input.columns:
                    game_df_input.loc[0, key] = value
                else:
                    print(f"Warning: Feature '{key}' from input dict not in model's expected features. It will be ignored.")

            # Ensure all expected columns are present, fill missing ones with NaN (imputer will handle)
            for col in all_feature_names_in_order:
                if col not in game_df_input.columns:
                    game_df_input[col] = np.nan # Should have been created by pd.DataFrame(columns=...)

            # Reorder to be absolutely sure, though pd.DataFrame(columns=...) should handle it
            game_df_input = game_df_input[all_feature_names_in_order]


            print(f"\nPredicting for new game with features: {features_dict}")
            if hasattr(pipeline_model, "predict_proba"):
                win_prob = pipeline_model.predict_proba(game_df_input)[0][1]
                print(f"  Win probability: {win_prob:.2%}")
                print(f"  Loss probability: {1-win_prob:.2%}")
                print(f"  Predicted outcome: {'Win' if win_prob > 0.5 else 'Loss'}")
                return win_prob
            else:
                prediction = pipeline_model.predict(game_df_input)[0]
                print(f"  Predicted outcome: {'Win' if prediction == 1 else 'Loss'} (probabilities not available)")
                return prediction

        print("\nTo predict a new game, use the predict_game_outcome function with the best model.")
        print("Example: feature_values = {'Feature1Name': value1, 'Feature2Name': value2, ...}")
        print("predict_game_outcome(best_model_pipeline, feature_values, all_selected_features)")
        # Example call (ensure `best_model_pipeline` and `all_selected_features` are defined)
        # if 'best_model_pipeline' in locals() and all_selected_features:
        #     sample_features_for_prediction = X_test.iloc[[0]].to_dict(orient='records')[0]
        #     predict_game_outcome(best_model_pipeline, sample_features_for_prediction, all_selected_features)


    else:
        print("No models were successfully trained or results dictionary is empty.")
