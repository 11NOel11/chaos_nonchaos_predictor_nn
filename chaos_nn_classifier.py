import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU before TensorFlow initializes

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Ensure TensorFlow does not use GPU
tf.config.set_visible_devices([], 'GPU')
tf.get_logger().setLevel('ERROR')

# Load dataset
df = pd.read_csv("chaotic_system_dataset.csv")

# Check for data leakage
print("Correlation Matrix:\n", df.corr())
print("Class distribution:\n", df.iloc[:, -1].value_counts())

# Remove highly correlated features (threshold > 0.9) to prevent redundancy
corr_matrix = df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
df = df.drop(columns=to_drop)

# Separate features and labels
X = df.iloc[:, :-1].values  # All columns except last
y = df.iloc[:, -1].values  # Last column (labels)

# Define stratified k-fold cross-validation
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
accuracies, precisions, recalls, f1_scores, all_y_true, all_y_pred = [], [], [], [], [], []
last_model, X_test_last, y_test_last = None, None, None

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Apply StandardScaler **inside the loop** to prevent data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    model = keras.Sequential([
        keras.layers.Dense(4, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(2, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=16, class_weight=class_weight_dict, verbose=0)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracies.append(accuracy)
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred_prob)
    
    last_model, X_test_last, y_test_last = model, X_test, y_test

# Compute and display average metrics
average_accuracy = np.mean(accuracies)
average_precision = np.mean(precisions)
average_recall = np.mean(recalls)
average_f1 = np.mean(f1_scores)

print(f"Average Test Accuracy: {average_accuracy * 100:.2f}%")
print(f"Average Precision: {average_precision * 100:.2f}%")
print(f"Average Recall: {average_recall * 100:.2f}%")
print(f"Average F1-Score: {average_f1 * 100:.2f}%")

# Lyapunov Exponent Baseline Comparison
lyapunov_threshold = 0  # Assume chaos when Lyapunov exponent > 0
lyapunov_predictions = (df.iloc[:, 0].values > lyapunov_threshold).astype(int)  # Using first column as Lyapunov exponent
lyapunov_accuracy = np.mean(lyapunov_predictions == y)

print(f"Baseline Lyapunov Exponent Accuracy: {lyapunov_accuracy * 100:.2f}%")

if average_accuracy > lyapunov_accuracy:
    print("Neural Network outperforms Lyapunov exponent classification.")
else:
    print("Lyapunov exponent classification is comparable or better.")

# ROC Curve Comparison
fpr, tpr, _ = roc_curve(all_y_true, all_y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Neural Network (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Chaos Classification')
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(all_y_true, (np.array(all_y_pred) > 0.5).astype(int))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Chaotic', 'Chaotic'], yticklabels=['Non-Chaotic', 'Chaotic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# SHAP Analysis
background = X_test_last[np.random.choice(X_test_last.shape[0], 100, replace=False)]
explainer = shap.KernelExplainer(last_model.predict, background)
shap_values = explainer.shap_values(X_test_last)

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_test_last, show=False)
plt.title("SHAP Summary Plot (Feature Importance)")
plt.show()

shap.dependence_plot(0, shap_values[0], X_test_last, show=False, alpha=0.5)
plt.title("SHAP Dependence Plot for Feature 0")
plt.show()
