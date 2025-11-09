# Liver Disease Classification - Machine Learning Assignment

## Project Overview

This project implements a **binary classification problem** to predict liver disease in patients using three different machine learning approaches:
1. **Logistic Regression**
2. **Hard Margin SVM (Support Vector Machine)**
3. **Soft Margin SVM**


## Dataset Information

**Dataset:** Indian Liver Patient Dataset (ILPD)  
**Source:** Kaggle 
**Total Samples:** 583 patients  
**Classes:**
- Liver Disease: 416 patients (71.4%)
- Healthy: 167 patients (28.6%)

### Features (10 Total):
1. **Age** - Patient's age in years
2. **Gender** - Male or Female
3. **Total Bilirubin** - Waste product from red blood cell breakdown (mg/dL)
4. **Direct Bilirubin** - Processed form of bilirubin (mg/dL)
5. **Alkaline Phosphotase** - Liver enzyme (IU/L)
6. **Alamine Aminotransferase (ALT)** - Liver enzyme (IU/L)
7. **Aspartate Aminotransferase (AST)** - Liver enzyme (IU/L)
8. **Total Proteins** - All proteins in blood (g/dL)
9. **Albumin** - Main protein made by liver (g/dL)
10. **Albumin/Globulin Ratio** - Balance of proteins

---

### Step 1: Import Libraries and Load Dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

**What happens:**
- Import necessary libraries for data manipulation, machine learning, and visualization
- Load the CSV file containing patient records
- Display basic information about the dataset (shape, statistics, missing values)

**Key Output:**
- 583 total samples
- 4 missing values found in Albumin/Globulin Ratio column
- Class distribution shows imbalanced data (more liver disease cases)

---

### Step 2: Data Preprocessing

```python
# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert Gender to numeric
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Convert target variable
df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
```

**What happens:**
- **Missing Values:** Fill the 4 missing values with the median of their respective columns
- **Gender Encoding:** Convert categorical 'Male'/'Female' to numeric 1/0
- **Target Conversion:** Convert labels from (1=disease, 2=healthy) to (1=disease, 0=healthy)

**Why this matters:**
- Machine learning algorithms require numeric data
- Missing values would cause errors
- Consistent labeling (1 for positive class, 0 for negative) is standard practice

---

### Step 3: Feature Preparation and Train-Test Split

```python
# Separate features and target
X = df.drop('Dataset', axis=1)
y = df['Dataset']

# Split data (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**What happens:**
- **Feature Separation:** Separate input features (X) from target label (y)
- **Train-Test Split:** Divide data into 70% for training (408 samples) and 30% for testing (175 samples)
- **Stratification:** Maintain the same class ratio in both training and testing sets
- **Feature Scaling:** Standardize features to have mean=0 and standard deviation=1

**Why this matters:**
- Testing on unseen data evaluates model generalization
- Stratification prevents biased splits in imbalanced datasets
- **Feature scaling is critical for SVM and Logistic Regression** because these algorithms are sensitive to feature magnitudes

---

### Step 4: Logistic Regression

```python
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
```

**What happens:**
- Train a Logistic Regression model on scaled training data
- Make predictions on the test set
- Evaluate performance using accuracy, precision, recall, and F1-score

**Results:**
- **Accuracy:** 73.71%
- **Precision:** 74.84%
- **Recall:** 95.20%
- **F1-Score:** 83.80%

**What this means:**
- The model correctly identifies 95.20% of actual liver disease cases (high recall)
- Of all patients predicted to have liver disease, 74.84% actually have it (precision)
- Good balance between precision and recall (F1-score of 83.80%)

---

### Step 5: Hard Margin SVM

```python
# Hard margin approximated by very large C value
hard_svm = SVC(kernel='linear', C=1e6, random_state=42)
hard_svm.fit(X_train_scaled, y_train)
y_pred_hard = hard_svm.predict(X_test_scaled)
```

**What happens:**
- Train a Hard Margin SVM by setting C=1,000,000 (extremely large)
- This forces the model to allow almost no misclassifications
- Find the hyperplane that maximally separates the two classes

**Results:**
- **Accuracy:** 70.29%
- **Precision:** 71.86%
- **Recall:** 96.00%
- **F1-Score:** 82.19%
- **Support Vectors:** 220 total (108 healthy, 112 disease)

**What this means:**
- Slightly lower accuracy than Logistic Regression
- Very high recall (96%) but lower precision
- Uses 220 support vectors (data points near the decision boundary)
- The strict margin constraint may cause overfitting to training data

---

### Step 6: Soft Margin SVM (Multiple C Values)

```python
C_values = [0.1, 1, 10, 100]

for C in C_values:
    soft_svm = SVC(kernel='linear', C=C, random_state=42)
    soft_svm.fit(X_train_scaled, y_train)
    # Evaluate...
```

**What happens:**
- Train multiple Soft Margin SVM models with different C values
- **C parameter** controls the trade-off between:
  - **Large C (e.g., 100):** Less tolerance for errors, stricter margin
  - **Small C (e.g., 0.1):** More tolerance for errors, wider margin

**Results (All C values showed similar performance):**
- **Accuracy:** 71.43%
- **Precision:** 71.43%
- **Recall:** 100% (detected ALL liver disease cases!)
- **F1-Score:** 83.33%

**What this means:**
- **Perfect recall (100%):** Every patient with liver disease was identified
- Lower precision means more false positives (healthy patients misclassified)
- The similarity across C values suggests the data has inherent class overlap
- More support vectors as C increases (less regularization)

---

### Step 7: Visualization and Comparison

```python
# Create comparison plots
- Accuracy comparison bar chart
- F1-Score comparison bar chart  
- Support Vectors comparison
- Precision vs Recall scatter plot
```

**What happens:**
- Generate comprehensive visualizations comparing all models
- Save as `model_comparison.png`
- Provide visual insights into model performance trade-offs

---

## Results Summary

### Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | Support Vectors |
|-------|----------|-----------|--------|----------|-----------------|
| **Logistic Regression** | 73.71% | 74.84% | 95.20% | **83.80%** | N/A |
| **Hard Margin SVM** | 70.29% | 71.86% | 96.00% | 82.19% | 220 |
| **Soft Margin SVM (C=0.1)** | 71.43% | 71.43% | **100%** | 83.33% | 260 |
| **Soft Margin SVM (C=1)** | 71.43% | 71.43% | **100%** | 83.33% | 259 |
| **Soft Margin SVM (C=10)** | 71.43% | 71.43% | **100%** | 83.33% | 273 |
| **Soft Margin SVM (C=100)** | 71.43% | 71.43% | **100%** | 83.33% | 278 |

---

## Key Findings

### 1. Overall Best Model: **Logistic Regression**
- **Highest accuracy** (73.71%)
- **Best precision** (74.84%) - fewer false positives
- Still maintains excellent recall (95.20%)
- **Best for balanced performance**

### 2. Hard Margin SVM Performance
- Achieved 70.29% accuracy
- High recall (96%) but lower precision (71.86%)
- **Struggled with data overlap** - liver disease data is not perfectly linearly separable
- Required 220 support vectors, indicating complex decision boundary
- **Real-world limitation:** Real medical data has noise and overlapping cases

### 3. Soft Margin SVM Performance
- **Perfect recall (100%)** across all C values - caught every disease case!
- Lower precision (71.43%) - more false alarms
- **Trade-off:** Prioritizes not missing disease cases over avoiding false positives
- Similar performance across C values suggests inherent data characteristics


## Conclusions

### Scientific Insights:
1. **No single "best" algorithm** - depends on priorities (recall vs precision)
2. **Feature scaling is crucial** - all models benefited from standardization
3. **Class imbalance affects performance** - 71% disease cases influenced predictions
4. **Real-world data has noise** - soft margin approaches are more practical

### Practical Takeaways:
- For **liver disease screening**: Use Soft Margin SVM (catches all cases)
- For **diagnostic confidence**: Use Logistic Regression (balanced performance)
- For **academic understanding**: Hard Margin SVM demonstrates theoretical concepts but has limited practical use


## How to Run This Project

### Requirements:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Execution:
```bash
python liver_disease_classification.py
```

### Output Files:
- `model_comparison.png` - Visual comparison of all models
- Console output with detailed metrics and analysis

---
