# Home Run Prediction Model - README

## Project Overview
This project contains three Jupyter notebooks for building, evaluating, and deploying a machine learning model to predict home runs using MLB Statcast data:

1. train_homerun_cleaned.ipynb - Model training notebook
2. eval_homerun.ipynb - Model evaluation notebook 
3. infer_homerun.ipynb - Inference notebook for predictions

## Requirements
bash
pip install pybaseball pandas numpy matplotlib seaborn scikit-learn pycaret[full] category_encoders xgboost lightgbm


---

## 1. Training Notebook (train_homerun_cleaned.ipynb)

### Purpose
Trains and saves an XGBoost model to predict home runs from Statcast data.

### Input Data
- Training dataset: [https://drive.google.com/file/d/1Fm4_CZm09p3rWCJCPsex-6-XOL9dDQq4/view?usp=sharing]
- Validation/Test dataset: [https://drive.google.com/file/d/18vp4EY8NbUvkIoP9jI7-mlST39akOwZb/view?usp=sharing]

### Key Features
- Data preprocessing pipeline
- Model comparison and selection
- Hyperparameter tuning
- Class imbalance handling

### Outputs
- Saved model: no_SMOTE_model_optimal_weights.pkl

### Execution
1. Set BASE_DIR in cell 2
2. Download and load the training and validation datasets
3. Run all cells sequentially

---

## 2. Evaluation Notebook (eval_homerun.ipynb)

### Purpose
Evaluates model performance on test/unseen data.

### Required Inputs
- Trained model file: [https://drive.google.com/file/d/1QB1q_LZobx9tqZ5PM0BeF2cVRcZe_fhK/view?usp=sharing]
- Test CSV file: [https://drive.google.com/file/d/1--2RIZp2Boh4G42fI5x5XK2Xv_TLZfrW/view?usp=sharing]

### Key Features
- Comprehensive performance metrics
- Confusion matrix visualization
- ROC and precision-recall curves
- Error analysis

### Execution
1. Download and set path to the trained model (.pkl file)
2. Download and provide path to test CSV file
3. Run all cells

---

## 3. Inference Notebook (infer_homerun.ipynb) 

### Purpose
Makes predictions on new data (1-5 rows).

### Required Inputs
- Trained model file: [https://drive.google.com/file/d/1QB1q_LZobx9tqZ5PM0BeF2cVRcZe_fhK/view?usp=sharing]
- Input CSV template: [https://drive.google.com/file/d/1--2RIZp2Boh4G42fI5x5XK2Xv_TLZfrW/view?usp=sharing]

### Key Features
- Processes individual predictions
- Shows prediction probabilities
- Visualizes influential features

### Execution
1. Download the trained model
2. Prepare input CSV using the template (1-5 rows)
3. Run notebook to get predictions

---

## Workflow
1. Train model using train_homerun_cleaned.ipynb with training data
2. Evaluate performance with eval_homerun.ipynb using test data
3. Make predictions on new data using infer_homerun.ipynb
