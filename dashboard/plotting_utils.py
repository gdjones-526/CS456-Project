"""Utility functions for generating plots and visualizations"""
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from django.core.files.base import ContentFile

def generate_predicted_vs_actual_plot(y_test, y_pred):
    """Generate predicted vs actual scatter plot for regression"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, s=20)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
    buffer.seek(0)
    plt.close()

    return ContentFile(buffer.read(), name='predicted_vs_actual.png')

def generate_confusion_matrix_plot(y_test, cm, max_classes=10):
    """Generate simplified confusion matrix for classification"""
    labels = np.unique(y_test)
    
    # Limit the number of classes displayed
    if len(labels) > max_classes:
        labels = labels[:max_classes]
        cm = cm[:max_classes, :max_classes]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                cbar=False)  # remove colorbar to simplify
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
    buffer.seek(0)
    plt.close()
    
    return ContentFile(buffer.read(), name='confusion_matrix.png')