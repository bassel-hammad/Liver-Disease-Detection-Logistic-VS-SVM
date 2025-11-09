"""
Liver Disease Classification
Assignment 4A - Machine Learning

This project implements classification of liver disease using:
1. Logistic Regression
2. Hard Margin SVM
3. Soft Margin SVM

Dataset: Indian Liver Patient Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("LIVER DISEASE CLASSIFICATION PROJECT")
print("="*60)
print()

# Step 1: Load the Dataset
print("Step 1: Loading Dataset...")
print("-" * 60)

# Load the dataset
df = pd.read_csv('indian_liver_patient.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  Total samples: {len(df)}")
print(f"  Total features: {len(df.columns) - 1}")
print()

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())
print()

# Dataset information
print("Dataset Information:")
print(df.info())
print()

# Statistical summary
print("Statistical Summary:")
print(df.describe())
print()

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())
print()

# Check class distribution
print("Class Distribution:")
print(df['Dataset'].value_counts())
print()
print("="*60)
print()

# Step 2: Data Preprocessing
print("Step 2: Data Preprocessing...")
print("-" * 60)

# Handle missing values (if any)
# The Albumin_and_Globulin_Ratio column often has missing values
if df.isnull().sum().sum() > 0:
    print(f"Found {df.isnull().sum().sum()} missing values")
    # Fill missing values with median
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("✓ Missing values filled with median")
else:
    print("✓ No missing values found")

# Convert Gender to numeric (Male=1, Female=0)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
print("✓ Gender converted to numeric (Male=1, Female=0)")

# Convert target variable
# Dataset column: 1 = Liver disease, 2 = No liver disease
# We'll convert to: 1 = Liver disease, 0 = No liver disease
df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
print("✓ Target variable converted (1=Disease, 0=Healthy)")

print()
print("Final Class Distribution:")
print(f"  Liver Disease (1): {(df['Dataset'] == 1).sum()}")
print(f"  Healthy (0): {(df['Dataset'] == 0).sum()}")
print()
print("="*60)
print()

# Step 3: Prepare Features and Target
print("Step 3: Preparing Features and Target...")
print("-" * 60)

# Separate features (X) and target (y)
X = df.drop('Dataset', axis=1)
y = df['Dataset']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print()

# Split data into training and testing sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print()

# Feature Scaling (important for SVM and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")
print()
print("="*60)
print()

# Step 4: Logistic Regression
print("Step 4: Logistic Regression")
print("-" * 60)

# Train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_log = log_reg.predict(X_test_scaled)

# Evaluate
accuracy_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

print("Logistic Regression Results:")
print(f"  Accuracy:  {accuracy_log:.4f}")
print(f"  Precision: {precision_log:.4f}")
print(f"  Recall:    {recall_log:.4f}")
print(f"  F1-Score:  {f1_log:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred_log, target_names=['Healthy', 'Liver Disease']))
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print()
print("="*60)
print()

# Step 5: Hard Margin SVM
print("Step 5: Hard Margin SVM (Linear Kernel, C=∞)")
print("-" * 60)

# Train Hard Margin SVM
# Hard margin is approximated by using very large C value
# C=1e6 (very large) means almost no tolerance for errors
hard_svm = SVC(kernel='linear', C=1e6, random_state=42)

try:
    hard_svm.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_hard = hard_svm.predict(X_test_scaled)
    
    # Evaluate
    accuracy_hard = accuracy_score(y_test, y_pred_hard)
    precision_hard = precision_score(y_test, y_pred_hard)
    recall_hard = recall_score(y_test, y_pred_hard)
    f1_hard = f1_score(y_test, y_pred_hard)
    
    print("Hard Margin SVM Results:")
    print(f"  Accuracy:  {accuracy_hard:.4f}")
    print(f"  Precision: {precision_hard:.4f}")
    print(f"  Recall:    {recall_hard:.4f}")
    print(f"  F1-Score:  {f1_hard:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred_hard, target_names=['Healthy', 'Liver Disease']))
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_hard))
    print()
    print(f"Number of Support Vectors: {hard_svm.n_support_}")
    
except Exception as e:
    print(f"⚠ Hard Margin SVM encountered an issue: {e}")
    print("This is expected if data is not perfectly linearly separable!")
    accuracy_hard = None

print()
print("="*60)
print()

# Step 6: Soft Margin SVM
print("Step 6: Soft Margin SVM (Linear Kernel, Various C values)")
print("-" * 60)

# Test different C values
C_values = [0.1, 1, 10, 100]
soft_svm_results = {}

for C in C_values:
    print(f"\nTesting C = {C}...")
    
    # Train Soft Margin SVM
    soft_svm = SVC(kernel='linear', C=C, random_state=42)
    soft_svm.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_soft = soft_svm.predict(X_test_scaled)
    
    # Evaluate
    accuracy_soft = accuracy_score(y_test, y_pred_soft)
    precision_soft = precision_score(y_test, y_pred_soft)
    recall_soft = recall_score(y_test, y_pred_soft)
    f1_soft = f1_score(y_test, y_pred_soft)
    
    # Store results
    soft_svm_results[C] = {
        'model': soft_svm,
        'predictions': y_pred_soft,
        'accuracy': accuracy_soft,
        'precision': precision_soft,
        'recall': recall_soft,
        'f1': f1_soft,
        'n_support': soft_svm.n_support_
    }
    
    print(f"  Accuracy:  {accuracy_soft:.4f}")
    print(f"  Precision: {precision_soft:.4f}")
    print(f"  Recall:    {recall_soft:.4f}")
    print(f"  F1-Score:  {f1_soft:.4f}")
    print(f"  Support Vectors: {soft_svm.n_support_}")

print()
print("="*60)
print()

# Step 7: Comparison and Visualization
print("Step 7: Model Comparison")
print("-" * 60)

# Create comparison table
print("\n{:<25} {:<12} {:<12} {:<12} {:<12}".format(
    "Model", "Accuracy", "Precision", "Recall", "F1-Score"
))
print("-" * 73)

print("{:<25} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
    "Logistic Regression", accuracy_log, precision_log, recall_log, f1_log
))

if accuracy_hard is not None:
    print("{:<25} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        "Hard Margin SVM", accuracy_hard, precision_hard, recall_hard, f1_hard
    ))

for C in C_values:
    results = soft_svm_results[C]
    print("{:<25} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        f"Soft Margin SVM (C={C})", 
        results['accuracy'], 
        results['precision'], 
        results['recall'], 
        results['f1']
    ))

print()
print("="*60)
print()

# Visualizations
print("Generating visualizations...")

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
models = ['Logistic\nRegression', 'Hard Margin\nSVM']
accuracies = [accuracy_log, accuracy_hard if accuracy_hard else 0]
models.extend([f'Soft SVM\n(C={C})' for C in C_values])
accuracies.extend([soft_svm_results[C]['accuracy'] for C in C_values])

colors = ['#1f77b4', '#ff7f0e'] + ['#2ca02c'] * len(C_values)
bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# Plot 2: F1-Score Comparison
ax2 = axes[0, 1]
f1_scores = [f1_log, f1_hard if accuracy_hard else 0]
f1_scores.extend([soft_svm_results[C]['f1'] for C in C_values])

bars = ax2.bar(models, f1_scores, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('F1-Score', fontsize=12)
ax2.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# Plot 3: Support Vectors for SVM models
ax3 = axes[1, 0]
svm_models = ['Hard\nMargin']
support_vectors = [sum(hard_svm.n_support_) if accuracy_hard else 0]
svm_models.extend([f'C={C}' for C in C_values])
support_vectors.extend([sum(soft_svm_results[C]['n_support']) for C in C_values])

colors_sv = ['#ff7f0e'] + ['#2ca02c'] * len(C_values)
bars = ax3.bar(svm_models, support_vectors, color=colors_sv, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Number of Support Vectors', fontsize=12)
ax3.set_title('Support Vectors in SVM Models', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=10)

# Plot 4: Precision vs Recall
ax4 = axes[1, 1]
precisions = [precision_log, precision_hard if accuracy_hard else 0]
recalls = [recall_log, recall_hard if accuracy_hard else 0]
precisions.extend([soft_svm_results[C]['precision'] for C in C_values])
recalls.extend([soft_svm_results[C]['recall'] for C in C_values])

model_names_short = ['Log Reg', 'Hard SVM'] + [f'Soft C={C}' for C in C_values]
ax4.scatter(recalls, precisions, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
for i, txt in enumerate(model_names_short):
    ax4.annotate(txt, (recalls[i], precisions[i]), fontsize=9, ha='center')
ax4.set_xlabel('Recall', fontsize=12)
ax4.set_ylabel('Precision', fontsize=12)
ax4.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'model_comparison.png'")
plt.show()

print()
print("="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print()
print("Key Findings:")
print("1. Logistic Regression: Probabilistic approach, good baseline")
print("2. Hard Margin SVM: Strict separation, may overfit or fail on noisy data")
print("3. Soft Margin SVM: Flexible, trades off margin size vs errors")
print("   - Small C: More tolerance for errors, wider margin")
print("   - Large C: Less tolerance for errors, closer to hard margin")
print()
print("✓ Check 'model_comparison.png' for visual comparisons")
print("="*60)
