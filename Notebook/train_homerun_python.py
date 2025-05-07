# %%
BASE_DIR = "/content"

# %% [markdown]
# # 1. MLB BASEBALL HOMERUN PREDICTION

# %%
# !pip install pybaseball

# %% [markdown]
# ## Data Fetching, Cleaning, PreProcessing  (uncomment to scrap data from 0)

# %%
# # Final with cleaning and replacing uids #may need t combine session wise file do not rely on combined file from this script

# import pandas as pd
# from pybaseball import statcast
# from tqdm import tqdm
# import time
# import numpy as np
# import os

# # ---- CONFIGURATION ----
# START_YEAR = 2018
# END_YEAR = 2023
# OUTPUT_DIR = BASE_DIR + "/statcast_data"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ---- FILTER CRITERIA ----
# INCLUDE_DESCRIPTIONS = [
#     'hit_into_play',
#     'hit_into_play_no_out',
#     'hit_into_play_score'
# ]

# EXCLUDE_EVENTS = [
#     'bunt_foul_tip',
#     'bunt_groundout',
#     'bunt_lineout',
#     'bunt_popout',
#     'bunt_foul',
#     'sac_bunt',
#     'sac_bunt_double_play',
#     'bunt_foul_tip',
#     'bunt_foul',
#     'missed_bunt',
#     'foul_bunt',
#     'foul_tip'
# ]

# # ---- COLUMN STRUCTURE ----
# FINAL_COLUMNS = [
#     'uid', 'home_team', 'sz_top', 'sz_bot', 'pitch_type',
#     'release_pos_x', 'release_pos_y', 'release_pos_z', 'stand', 'p_throws',
#     'inning', 'inning_topbot', 'outs_when_up', 'balls', 'strikes',
#     'pitch_number', 'at_bat_number', 'if_fielding_alignment', 'of_fielding_alignment',
#     'on_3b', 'on_2b', 'on_1b', 'release_speed', 'spin_axis', 'release_spin_rate',
#     'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'hc_x', 'hc_y',
#     'launch_speed', 'launch_angle', 'is_hr'
# ]

# def get_monthly_date_ranges(year):
#     """Generate first-last day of each month for filtering"""
#     return [(f"{year}-{month:02d}-01",
#              f"{year}-{month:02d}-{28 if month == 2 else 30}")
#             for month in range(1, 13)]

# def fetch_batted_balls(start_date, end_date):
#     """Fetch only batted balls that match our criteria"""
#     for attempt in range(3):
#         try:
#             # First get all data for the date range
#             data = statcast(start_dt=start_date, end_dt=end_date)

#             if data.empty:
#                 return pd.DataFrame()

#             # Apply filters
#             filtered = data[
#                 (data['description'].isin(INCLUDE_DESCRIPTIONS)) &
#                 (~data['events'].isin(EXCLUDE_EVENTS)) &
#                 (data['hc_x'].notna()) &  # Must have hit coordinates
#                 (data['hc_y'].notna())
#             ]

#             return filtered

#         except Exception as e:
#             print(f"Attempt {attempt+1} failed: {str(e)}")
#             time.sleep(5)
#     return pd.DataFrame()

# def transform_to_final_format(raw_df):
#     """Convert raw data to exact desired format"""
#     if raw_df.empty:
#         return pd.DataFrame(columns=FINAL_COLUMNS)

#     # Create new DataFrame with our structure
#     df = pd.DataFrame()

#     # Direct mapping columns
#     direct_map = {
#         'game_pk': 'uid',
#         'home_team': 'home_team',
#         'sz_top': 'sz_top',
#         'sz_bot': 'sz_bot',
#         'pitch_type': 'pitch_type',
#         'release_pos_x': 'release_pos_x',
#         'release_pos_y': 'release_pos_y',
#         'release_pos_z': 'release_pos_z',
#         'stand': 'stand',
#         'p_throws': 'p_throws',
#         'inning': 'inning',
#         'inning_topbot': 'inning_topbot',
#         'outs_when_up': 'outs_when_up',
#         'balls': 'balls',
#         'strikes': 'strikes',
#         'pitch_number': 'pitch_number',
#         'at_bat_number': 'at_bat_number',
#         'if_fielding_alignment': 'if_fielding_alignment',
#         'of_fielding_alignment': 'of_fielding_alignment',
#         'release_speed': 'release_speed',
#         'spin_axis': 'spin_axis',
#         'release_spin_rate': 'release_spin_rate',
#         'pfx_x': 'pfx_x',
#         'pfx_z': 'pfx_z',
#         'plate_x': 'plate_x',
#         'plate_z': 'plate_z',
#         'hc_x': 'hc_x',
#         'hc_y': 'hc_y',
#         'launch_speed': 'launch_speed',
#         'launch_angle': 'launch_angle'
#     }

#     for src, dest in direct_map.items():
#         if src in raw_df.columns:
#             df[dest] = raw_df[src]

#     # Handle boolean columns for base runners
#     for base in ['1b', '2b', '3b']:
#         df[f'on_{base}'] = raw_df[f'on_{base}'].notna()

#     # Create is_hr column
#     df['is_hr'] = (raw_df['events'] == 'home_run').astype(int)

#     # Ensure all columns are present
#     for col in FINAL_COLUMNS:
#         if col not in df.columns:
#             df[col] = None

#     return df[FINAL_COLUMNS]

# def process_year(year):
#     """Process a single year of data"""
#     all_data = []
#     date_ranges = get_monthly_date_ranges(year)

#     print(f"\nDownloading batted balls for {year} ({len(date_ranges)} months)...")
#     for start_date, end_date in tqdm(date_ranges, desc=f"{year}"):
#         raw_data = fetch_batted_balls(start_date, end_date)
#         if not raw_data.empty:
#             processed = transform_to_final_format(raw_data)
#             all_data.append(processed)
#         time.sleep(2)  # Be kind to MLB servers

#     if not all_data:
#         print(f"No data retrieved for {year}!")
#         return None

#     final_df = pd.concat(all_data).sort_values('uid').reset_index(drop=True)
#     return final_df

# def clean_and_renumber_data(yearly_data):
#     """
#     NEW FUNCTION: Processes data before saving to:
#     1. Drop rows with any missing values
#     2. Assign sequential UIDs (0 to N-1)
#     Returns cleaned yearly DataFrames
#     """
#     # Combine all data for cleaning
#     combined = pd.concat([df for df in yearly_data.values() if df is not None])

#     # Drop rows with any missing values
#     cleaned = combined.dropna(how='any')
#     print(f"\nData cleaning:")
#     print(f" - Original rows: {len(combined)}")
#     print(f" - Rows kept: {len(cleaned)} ({len(cleaned)/len(combined):.1%})")

#     # Assign new sequential UIDs
#     cleaned['uid'] = range(len(cleaned))

#     # Split back into yearly DataFrames
#     cleaned_yearly = {}
#     start_idx = 0
#     for year, df in yearly_data.items():
#         if df is not None:
#             end_idx = start_idx + len(df.dropna(how='any'))
#             cleaned_yearly[year] = cleaned.iloc[start_idx:end_idx].copy()
#             start_idx = end_idx

#     return cleaned_yearly

# def main():
#     yearly_data = {}

#     # Step 1: Collect raw data (original process)
#     for year in range(START_YEAR, END_YEAR + 1):
#         yearly_data[year] = process_year(year)

#     # Step 2: Clean and renumber (new process)
#     yearly_data = clean_and_renumber_data(yearly_data)

#     # Step 3: Save files (original process)
#     for year, df in yearly_data.items():
#         if df is not None:
#             output_file = os.path.join(OUTPUT_DIR, f"statcast_batted_balls_{year}.csv")
#             df.to_csv(output_file, index=False)
#             print(f"Saved {len(df)} rows to {output_file}")

#     # Save combined file
#     combined_df = pd.concat([df for df in yearly_data.values() if df is not None])
#     combined_file = os.path.join(OUTPUT_DIR, "statcast_batted_balls_ALL.csv")
#     combined_df.to_csv(combined_file, index=False)
#     print(f"\nCombined all years into {combined_file} ({len(combined_df)} rows)")
#     print(f"UID range: {combined_df['uid'].min()} to {combined_df['uid'].max()}")

# if __name__ == "__main__":
#     main()

# %%
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # import os

# # input_path = '/content/statcast_data'  # Directory containing CSV files
# # multiple_files = True  # False  # Set to True if using multiple files

# # # Load dataset
# # def readData():
# #     if multiple_files:
# #         # Read all CSV files in directory
# #         print(f"Looking for CSV files in: {input_path}")
# #         all_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
# #                     if f.endswith('.csv')]
# #         print(f"Found {len(all_files)} CSV files")

# #         # Combine all files
# #         data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
# #         print(f"Combined {len(all_files)} files into DataFrame")
# #         return data
# #     else:
# #         # Read single file
# #         data = pd.read_csv('/content/statcast_batted_balls_2022.csv')
# #         print("Loaded single CSV file")
# #         return data

# # data = readData()
# # #check and drop missing vals
# # original_row_count = len(data)
# # # Drop rows with any missing values
# # data = data.dropna()
# # # Calculate percentage of rows removed
# # dropped_percentage = (original_row_count - len(data)) / original_row_count
# # # Raise error if more than 40% dropped
# # if dropped_percentage > 0.40:
# #     raise ValueError(f"Too much data removed! Dropped {dropped_percentage:.2%} of rows.")


# # # Split into train and test (80% train, 20% test by default)
# # train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# # # Save to CSV files
# # train_data.to_csv('/content/statcast_data/train.csv', index=False)
# # test_data.to_csv('/content/statcast_data/b_test.csv', index=False)

# # print(f"Train set shape: {train_data.shape}")
# # print(f"Test set shape: {test_data.shape}")
# # print("Files saved as 'train.csv' and 'test.csv'")

# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split

# input_path = BASE_DIR + "/statcast_data"  # Directory containing CSV files (must have no missing values)

# csv_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.csv')]
# print(f"Found {len(csv_files)} CSV files to merge.")

# df_list = []
# for idx, file in enumerate(csv_files, 1):
#     print(f"Reading file {idx}/{len(csv_files)}: {os.path.basename(file)}")
#     df = pd.read_csv(file)
#     df_list.append(df)

# print("Merging all files into a single DataFrame...")
# data = pd.concat(df_list, ignore_index=True)
# data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# print("Checking for missing values...")
# if data.isnull().any().any():
#     missing_rows = data[data.isnull().any(axis=1)]
#     print(f"Error: Found {len(missing_rows)} rows with missing values.")
#     print(missing_rows.head())
#     raise ValueError("Missing value rows found before saving.")

# print("Splitting into train and test sets...")
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# train_path = os.path.join(BASE_DIR, 'b_train.csv')
# test_path = os.path.join(BASE_DIR, 'b_test.csv')
# train_data.to_csv(train_path, index=False)
# test_data.to_csv(test_path, index=False)

# print(f"Train set shape: {train_data.shape}")
# print(f"Test set shape: {test_data.shape}")
# print(f"Files saved as '{train_path}' and '{test_path}'")

# %% [markdown]
# ## DATA LOADING, Checking & Encoding

# %%
# !pip install --ignore-installed blinker --upgrade
# !pip install pycaret[full]

# %%
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from pycaret.classification import setup


train1=pd.read_csv(BASE_DIR + "/b_train.csv")
test1=pd.read_csv(BASE_DIR + "/b_test.csv")

# train1=pd.read_csv(BASE_DIR + "/b_train.csv").head(35000)
# test1=pd.read_csv(BASE_DIR + "/b_test.csv").head(8000)
train1.head()

# %%
train1.isnull().any().sum(),test1.isnull().any().sum()

# %%
def find(null):
    sum=pd.DataFrame(null.dtypes,columns=['data types'])
    sum['missing']=null.isnull().sum()
    sum['unique']=null.nunique()
    return sum
find(train1).style.background_gradient(cmap='tab20')

# %%
def fine(nulls):
    sum=pd.DataFrame(nulls.dtypes,columns=['data types'])
    sum['missing']=nulls.isnull().sum()
    sum['unique']=nulls.nunique()
    return sum
fine(test1).style.background_gradient(cmap='tab20')

# %%
plt.figure(facecolor="#f5a2ee")
train1.dtypes.value_counts().plot(kind='pie')


plt.figure(facecolor="#f5a2ee")
test1.dtypes.value_counts().plot(kind='pie')

plt.figure(facecolor="#f5a2ee")
sns.countplot(data=train1)

plt.figure(facecolor="#f5a2ee",figsize=(10,5))
# print(test.dtypes)
numeric_df = test1.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(),annot=True)

# %%
train1.columns,test1.columns

# %% [markdown]
# ## Experiments & Selection

# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pycaret.classification import *

train=train1
test=test1

# Verify train and test columns for consistency
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

# Label encoding for categorical features (excluding high-cardinality and boolean) (CONFIG-2)
categorical_cols = ['stand', 'p_throws', 'inning_topbot', 'if_fielding_alignment', 'of_fielding_alignment']
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on union of train and test to ensure consistent encoding
    le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# One-hot encoding for high-cardinality features: home_team, pitch_type
train = pd.get_dummies(train, columns=['home_team', 'pitch_type'], prefix=['home_team', 'pitch_type'])
test = pd.get_dummies(test, columns=['home_team', 'pitch_type'], prefix=['home_team', 'pitch_type'])
# Align train and test columns after one-hot encoding
train, test = train.align(test, join='left', axis=1, fill_value=0)

# Skip encoding for boolean columns (on_3b, on_2b, on_1b) as they're already 0/1
# Convert booleans to int for consistency
for col in ['on_3b', 'on_2b', 'on_1b']:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

# Enhanced EDA: Data type distribution
plt.figure(figsize=(10, 5), facecolor="#f5a2ee")
train.dtypes.value_counts().plot(kind='pie', cmap="rainbow_r", autopct='%1.1f%%', title="Data Types Distribution")
plt.show()

# Enhanced EDA: Correlation heatmap focused on is_hr
plt.figure(figsize=(10, 8), facecolor="#f5a2ee")
# Select numeric columns for correlation
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
corr = train[numeric_cols].corr()[['is_hr']].sort_values(by='is_hr', ascending=False)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, vmin=-1, vmax=1)
plt.title("Correlation with Home Run (is_hr)")
plt.show()

# Enhanced EDA: Distribution of key features by is_hr
for feature in ['launch_speed', 'launch_angle']:
    plt.figure(figsize=(10, 5), facecolor="#f5a2ee")
    sns.histplot(data=train, x=feature, hue='is_hr', multiple='stack', bins=30)
    plt.title(f"Distribution of {feature} by Home Run")
    plt.show()

# PyCaret setup with fixes for class imbalance and reduced outlier removal
clf = setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    outliers_threshold=0.01,  # Lowered threshold to preserve more data
    fix_imbalance=True,  # Address class imbalance with SMOTE or similar
    session_id=1921,  # Keep same session ID for reproducibility
    verbose=True
)
models()

# Compare models, sort by F1 to prioritize minority class performance
compare_models(sort='F1', verbose=True)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from pycaret.classification import *
train=train1
test=test1

# Verify train and test columns
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

# Label encoding for low-cardinality categorical features (CONFIG-3)
categorical_cols = ['stand', 'p_throws', 'inning_topbot', 'if_fielding_alignment', 'of_fielding_alignment']
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Target encoding for high-cardinality features to avoid feature explosion
te = TargetEncoder(cols=['home_team', 'pitch_type'])
train = te.fit_transform(train, train['is_hr'])
test = te.transform(test)

# Convert boolean columns to int
for col in ['on_3b', 'on_2b', 'on_1b']:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

# EDA: Data type distribution
plt.figure(figsize=(10, 5), facecolor="#f5a2ee")
train.dtypes.value_counts().plot(kind='pie', cmap="rainbow_r", autopct='%1.1f%%', title="Data Types Distribution")
plt.show()

# EDA: Correlation heatmap for is_hr
plt.figure(figsize=(10, 8), facecolor="#f5a2ee")
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
corr = train[numeric_cols].corr()[['is_hr']].sort_values(by='is_hr', ascending=False)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, vmin=-1, vmax=1)
plt.title("Correlation with Home Run (is_hr)")
plt.show()

# EDA: Distribution of key features
for feature in ['launch_speed', 'launch_angle']:
    plt.figure(figsize=(10, 5), facecolor="#f5a2ee")
    sns.histplot(data=train, x=feature, hue='is_hr', multiple='stack', bins=30)
    plt.title(f"Distribution of {feature} by Home Run")
    plt.show()



# PyCaret setup with imbalance handling and minimal outlier removal
clf = setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    outliers_threshold=0.01,
    fix_imbalance=True,
    fix_imbalance_method='smote',
    session_id=1921,
    verbose=True
)

# Compare models by F1 score
compare_models(sort='F1', verbose=True)

# %% [markdown]
# SMOTE V/S Without SMOTE

# %%
from pycaret.classification import*
clf = setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    outliers_threshold=0.01,
    fix_imbalance=True,
    fix_imbalance_method='smote',
    session_id=1921,
    verbose=True
)
compare_models(
    include=['xgboost', 'lightgbm', 'gbc', 'ada'],
    sort='F1',
    verbose=True
)

# %%
clf = setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    fix_imbalance=True,
    session_id=1921,
    verbose=True
)

best_model = compare_models(
    include=['xgboost', 'lightgbm', 'gbc', 'ada'],
    sort='F1',
    verbose=True
)

# %% [markdown]
# Different configurations for model

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from pycaret.classification import *
from sklearn.metrics import f1_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Verify train and test DataFrames
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Preprocessing
def preprocess_data(train, test):
    print("Applying preprocessing...")
    # Label encoding for low-cardinality categorical features
    categorical_cols = ['stand', 'p_throws', 'inning_topbot', 'if_fielding_alignment', 'of_fielding_alignment']
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    # Target encoding for high-cardinality features
    te = TargetEncoder(cols=['home_team', 'pitch_type'])
    train = te.fit_transform(train, train['is_hr'])
    test = te.transform(test)

    # Convert boolean columns to int
    for col in ['on_3b', 'on_2b', 'on_1b']:
        train[col] = train[col].astype(int)
        test[col] = test[col].astype(int)

    return train, test

# Apply preprocessing (uncomment if your data isn't preprocessed)
# train, test = preprocess_data(train, test)

# Initialize results list
results = []

# Experiment 1: Baseline (No Imbalance Handling)
print("\nExperiment 1: Baseline (No Imbalance Handling)")
clf1 = setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    outliers_threshold=0.01,
    fix_imbalance=False,
    session_id=1921,
    verbose=False
)
for model_name in ['xgboost', 'lightgbm']:
    try:
        model = create_model(model_name)
        pred_df = predict_model(model, data=test)
        y_pred = pred_df['prediction_label']  # Updated to use 'prediction_label'
        f1 = f1_score(test['is_hr'], y_pred)
        recall = recall_score(test['is_hr'], y_pred)
        results.append({'Experiment': 'Baseline', 'Model': model_name, 'F1': f1, 'Recall': recall})
        print(f"{model_name} - F1: {f1:.4f}, Recall: {recall:.4f}")
    except Exception as e:
        print(f"Error in {model_name}: {str(e)}")

# Experiment 2: Class Weights
print("\nExperiment 2: Class Weights")
pos_weight = sum(train['is_hr'] == 0) / sum(train['is_hr'] == 1)  # Ratio of negative to positive
clf2 = setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    outliers_threshold=0.01,
    fix_imbalance=False,
    session_id=1921,
    verbose=False
)
models = {
    'xgboost': XGBClassifier(scale_pos_weight=pos_weight, use_label_encoder=False, eval_metric='logloss', random_state=1921),
    'lightgbm': LGBMClassifier(scale_pos_weight=pos_weight, random_state=1921)
}
for model_name, model in models.items():
    try:
        model.fit(train.drop(['is_hr', 'uid'], axis=1), train['is_hr'])
        y_pred = model.predict(test.drop(['is_hr', 'uid'], axis=1))
        f1 = f1_score(test['is_hr'], y_pred)
        recall = recall_score(test['is_hr'], y_pred)
        results.append({'Experiment': 'Class Weights', 'Model': model_name, 'F1': f1, 'Recall': recall})
        print(f"{model_name} - F1: {f1:.4f}, Recall: {recall:.4f}")
    except Exception as e:
        print(f"Error in {model_name}: {str(e)}")

# Experiment 3: SMOTE
print("\nExperiment 3: SMOTE")
clf3 = setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    outliers_threshold=0.01,
    fix_imbalance=True,
    fix_imbalance_method='smote',
    session_id=1921,
    verbose=False
)
for model_name in ['xgboost', 'lightgbm']:
    try:
        model = create_model(model_name)
        pred_df = predict_model(model, data=test)
        y_pred = pred_df['prediction_label']  # Updated to use 'prediction_label'
        f1 = f1_score(test['is_hr'], y_pred)
        recall = recall_score(test['is_hr'], y_pred)
        results.append({'Experiment': 'SMOTE', 'Model': model_name, 'F1': f1, 'Recall': recall})
        print(f"{model_name} - F1: {f1:.4f}, Recall: {recall:.4f}")
    except Exception as e:
        print(f"Error in {model_name}: {str(e)}")

# Experiment 4: SMOTE + Tomek Links
print("\nExperiment 4: SMOTE + Tomek Links")
smotetomek = SMOTETomek(random_state=1921)
X_train, y_train = smotetomek.fit_resample(train.drop(['is_hr', 'uid'], axis=1), train['is_hr'])
train_balanced = pd.concat([X_train, y_train], axis=1)
clf4 = setup(
    data=train_balanced,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    outliers_threshold=0.01,
    fix_imbalance=False,
    session_id=1921,
    verbose=False
)
for model_name in ['xgboost', 'lightgbm']:
    try:
        model = create_model(model_name)
        pred_df = predict_model(model, data=test)
        y_pred = pred_df['prediction_label']  # Updated to use 'prediction_label'
        f1 = f1_score(test['is_hr'], y_pred)
        recall = recall_score(test['is_hr'], y_pred)
        results.append({'Experiment': 'SMOTE+Tomek', 'Model': model_name, 'F1': f1, 'Recall': recall})
        print(f"{model_name} - F1: {f1:.4f}, Recall: {recall:.4f}")
    except Exception as e:
        print(f"Error in {model_name}: {str(e)}")

# Experiment 5: Regularization
print("\nExperiment 5: Regularization")
clf5 = setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    outliers_threshold=0.01,
    fix_imbalance=True,
    session_id=1921,
    verbose=False
)
models_reg = {
    'xgboost': XGBClassifier(alpha=1.0, lambda_=1.0, max_depth=5, use_label_encoder=False, eval_metric='logloss', random_state=1921),
    'lightgbm': LGBMClassifier(reg_alpha=1.0, reg_lambda=1.0, max_depth=5, random_state=1921)
}
for model_name, model in models_reg.items():
    try:
        model.fit(train.drop(['is_hr', 'uid'], axis=1), train['is_hr'])
        y_pred = model.predict(test.drop(['is_hr', 'uid'], axis=1))
        f1 = f1_score(test['is_hr'], y_pred)
        recall = recall_score(test['is_hr'], y_pred)
        results.append({'Experiment': 'Regularization', 'Model': model_name, 'F1': f1, 'Recall': recall})
        print(f"{model_name} - F1: {f1:.4f}, Recall: {recall:.4f}")
    except Exception as e:
        print(f"Error in {model_name}: {str(e)}")

# Experiment 6: Hyperparameter Tuning
print("\nExperiment 6: Hyperparameter Tuning")
clf6 = setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    outliers_threshold=0.01,
    fix_imbalance=True,
    session_id=1921,
    verbose=False
)
for model_name in ['xgboost', 'lightgbm']:
    try:
        model = create_model(model_name)
        tuned_model = tune_model(model, optimize='F1', n_iter=10, search_library='scikit-optimize')
        pred_df = predict_model(tuned_model, data=test)
        y_pred = pred_df['prediction_label']  # Updated to use 'prediction_label'
        f1 = f1_score(test['is_hr'], y_pred)
        recall = recall_score(test['is_hr'], y_pred)
        results.append({'Experiment': 'Tuning', 'Model': model_name, 'F1': f1, 'Recall': recall})
        print(f"{model_name} - F1: {f1:.4f}, Recall: {recall:.4f}")
    except Exception as e:
        print(f"Error in {model_name}: {str(e)}")

# Experiment 7: Feature Selection
print("\nExperiment 7: Feature Selection")
clf7 = setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    outliers_threshold=0.01,
    fix_imbalance=True,
    session_id=1921,
    verbose=False
)
for model_name in ['xgboost', 'lightgbm']:
    try:
        model = create_model(model_name)
        # Generate feature importance plot
        plot_model(model, plot='feature', save=True)
        # Select top 15 features (adjust based on importance plot)
        top_features = get_config('X_train').columns[:15]
        train_selected = train[list(top_features) + ['is_hr']]
        test_selected = test[list(top_features) + ['is_hr']]
        clf7_selected = setup(
            data=train_selected,
            target='is_hr',
            normalize=True,
            normalize_method='robust',
            remove_outliers=True,
            outliers_threshold=0.01,
            fix_imbalance=True,
            session_id=1921,
            verbose=False
        )
        model_selected = create_model(model_name)
        pred_df = predict_model(model_selected, data=test_selected)
        y_pred = pred_df['prediction_label']  # Updated to use 'prediction_label'
        f1 = f1_score(test_selected['is_hr'], y_pred)
        recall = recall_score(test_selected['is_hr'], y_pred)
        results.append({'Experiment': 'Feature Selection', 'Model': model_name, 'F1': f1, 'Recall': recall})
        print(f"{model_name} - F1: {f1:.4f}, Recall: {recall:.4f}")
    except Exception as e:
        print(f"Error in {model_name}: {str(e)}")

# Summarize and visualize results
results_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(results_df)

# Bar plot for F1 scores
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='F1', hue='Experiment', data=results_df)
plt.title('F1 Score by Model and Experiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar plot for Recall
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Recall', hue='Experiment', data=results_df)
plt.title('Recall by Model and Experiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save the best model
best_result = results_df.loc[results_df['F1'].idxmax()]
best_model_name = best_result['Model']
best_experiment = best_result['Experiment']
print(f"\nBest Model: {best_model_name} from {best_experiment} with F1: {best_result['F1']:.4f}")
clf_final = setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    outliers_threshold=0.01,
    fix_imbalance=(best_experiment in ['SMOTE', 'SMOTE+Tomek']),
    session_id=1921,
    verbose=False
)
# final_model = create_model(best_model_name)
# save_model(final_model, 'best_home_run_model')

# %%
from pycaret.classification import *

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# train=train1
# test=test1
# Step 1: Visualize Original Feature Space (Before PCA)
# Assuming `train` is your original dataset
X = train.drop(columns=["is_hr", "uid"])  # Drop target and non-numeric columns
y = train["is_hr"]  # Target variable

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a scatter plot for original data (First two features for simplicity)
plt.figure(figsize=(10,6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y, palette='Set1')
plt.title("Feature Space - Original Data (Before PCA)")
plt.show()


# ------------------------------
# Uncomment this block for **no PCA** setup
# ------------------------------
# setup(train, target="is_hr", ignore_features="uid",
#       normalize_method="robust", remove_outliers=True)  # No PCA


# %%
# ------------------------------
# Uncomment this block for **PCA** setup
# ------------------------------

# Step 1: Apply PCA during setup
setup(train, target="is_hr", ignore_features="uid",
      normalize_method="robust", remove_outliers=True,
      pca=True, pca_components=0.99)  # Enable PCA for dimensionality reduction

# Step 2: Retrieve the transformed data after PCA
X_transformed = get_config('X_train')  # This gives the PCA-transformed dataset

# Step 3: Plot the first two PCA components
# Use iloc to select columns correctly
plt.figure(figsize=(10,6))
sns.scatterplot(x=X_transformed.iloc[:, 0], y=X_transformed.iloc[:, 1], hue=train['is_hr'], palette='Set1')
plt.title("Feature Space - After PCA (First Two Components)")
plt.show()

# %%


# ------------------------------
# Experiment 1: Ridge Classifier with Class Weights
# ------------------------------
print("---- Experiment 1: Ridge Classifier with Class Weights ----")
ridge_model = create_model('ridge', class_weight='balanced')  # Ridge with balanced class weights
tuned_ridge = tune_model(ridge_model)
print(tuned_ridge)
print("\n" + "-"*80)  # Separator for readability

# ------------------------------
# Experiment 2: Logistic Regression with Class Weights
# ------------------------------
print("---- Experiment 2: Logistic Regression with Class Weights ----")
lr_model = create_model('lr', class_weight='balanced')  # Logistic Regression with balanced class weights
tuned_lr = tune_model(lr_model)
print(tuned_lr)
print("\n" + "-"*80)  # Separator for readability

# ------------------------------
# Experiment 3: XGBoost Classifier with Class Weights
# ------------------------------
print("---- Experiment 4: XGBoost Classifier with Class Weights ----")
xgboost_model = create_model('xgboost', class_weight='balanced')  # XGBoost with balanced class weights
tuned_xgboost = tune_model(xgboost_model)
print(tuned_xgboost)
print("\n" + "-"*80)  # Separator for readability



# %%
cols_to_remove = ['hc_x', 'hc_y', 'launch_speed', 'launch_angle']
setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'] + cols_to_remove,
    fix_imbalance=True,  # Enable SMOTE
    remove_multicollinearity=True,
    feature_selection=True,
    normalize=True,
    session_id=42
)

models()
compare_models()

# %% [markdown]
# ## Final Model and Training

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from pycaret.classification import *
train=train1
test=test1

# Verify train and test columns
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

# Label encoding for low-cardinality categorical features (CONFIG-3)
categorical_cols = ['stand', 'p_throws', 'inning_topbot', 'if_fielding_alignment', 'of_fielding_alignment']
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Target encoding for high-cardinality features to avoid feature explosion
te = TargetEncoder(cols=['home_team', 'pitch_type'])
train = te.fit_transform(train, train['is_hr'])
test = te.transform(test)

# Convert boolean columns to int
for col in ['on_3b', 'on_2b', 'on_1b']:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)


# %%
import joblib

# Save the fitted target encoder
joblib.dump(te, 'target_encoder.pkl')


# %%
plt.figure(facecolor="#f5a2ee",figsize=(10,5))
sns.heatmap(train.corr(),annot=False)

# %%
from pycaret.classification import *
import os
import warnings
warnings.filterwarnings('ignore')

clf = setup(
    data=train,
    target='is_hr',
    ignore_features=['uid'],
    normalize=True,
    normalize_method='robust',
    remove_outliers=True,
    outliers_threshold=0.01,
    fold=10,  # 10-fold cross-validation
    session_id=1921,
    verbose=True
)

# Create XGBoost model
xgb_model = create_model('xgboost', fold=10)

# Tune XGBoost model with early stopping (PyCaret >= 3.0.0)
tuned_xgb = tune_model(
    xgb_model,
    optimize='F1',
    fold=10,
    n_iter=25,
    early_stopping=True,
    early_stopping_max_iters=10,  # early_stopping_rounds
    choose_better=True
)

# %%
# Evaluate on test set
preds = predict_model(tuned_xgb, data=test)
print(preds[['is_hr', 'prediction_label', 'prediction_score']].head())

# Save the best model to BASE_DIR
model_save_path = os.path.join(BASE_DIR, "no_SMOTE_model_optimal_weights")
save_model(tuned_xgb, model_save_path)
print(f"Model saved to {model_save_path}.pkl")
best = tuned_xgb

# %%
from pycaret.classification import plot_model
import matplotlib.pyplot as plt

# --- OR: Custom Plot for Top 20 Features ---
# Extract feature importance
importance = best.feature_importances_
features = best.get_booster().feature_names

# Sort and select top 20
sorted_idx = importance.argsort()[-20:][::-1]  # Get top 20 indices
top_features = [features[i] for i in sorted_idx]
top_importance = importance[sorted_idx]

# Plot manually
plt.figure(figsize=(10, 8))
plt.barh(top_features, top_importance, color='#1f77b4')
plt.xlabel('Importance')
plt.title('Top 20 Features for XGBoost')
plt.gca().invert_yaxis()  # Highest importance at top
plt.show()

# %%
plot_model(best,plot="auc")
plot_model(best,plot="error")
plot_model(best,plot='confusion_matrix', plot_kwargs = {'percent' : True})
plot_model(best,plot="class_report") # error here

# %%




