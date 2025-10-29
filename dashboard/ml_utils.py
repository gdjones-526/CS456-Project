"""
Machine Learning training utilities for scikit-learn models
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from django.core.files.base import ContentFile


class ModelTrainer:
    """Handles training and evaluation of ML models"""
    
    ALGORITHMS = {
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegression,
        'svm': SVC,
        'neural_network': MLPClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'decision_tree': DecisionTreeClassifier,
    }
    
    def __init__(self, dataset_path, target_column, algorithm='random_forest', 
                 test_size=0.2, random_state=42, algorithm_params=None):
        """
        Initialize the trainer
        
        Args:
            dataset_path: Path to the dataset file
            target_column: Name of the target variable column
            algorithm: Algorithm to use for training
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            algorithm_params: Additional parameters for the algorithm
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.algorithm = algorithm
        self.test_size = test_size
        self.random_state = random_state
        self.algorithm_params = algorithm_params or {}
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.label_encoders = {}
        
    def load_and_preprocess_data(self):
        """Load dataset and perform preprocessing"""
        # Load data based on file extension
        ext = os.path.splitext(self.dataset_path)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(self.dataset_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(self.dataset_path)
        elif ext == '.json':
            df = pd.read_json(self.dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Check if target column exists
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))
        
        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders['target'] = le
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return True
    
    def train_model(self):
        """Train the selected model"""
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first")
        
        # Get the model class
        model_class = self.ALGORITHMS.get(self.algorithm)
        if not model_class:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Set default parameters based on algorithm
        default_params = self._get_default_params()
        params = {**default_params, **self.algorithm_params}
        
        # Initialize and train model
        self.model = model_class(**params)
        self.model.fit(self.X_train, self.y_train)
        
        return self.model
    
    def _get_default_params(self):
        """Get default parameters for each algorithm"""
        defaults = {
            'random_forest': {
                'n_estimators': 100,
                'random_state': self.random_state,
                'max_depth': 10,
            },
            'logistic_regression': {
                'random_state': self.random_state,
                'max_iter': 1000,
            },
            'svm': {
                'random_state': self.random_state,
                'kernel': 'rbf',
            },
            'neural_network': {
                'random_state': self.random_state,
                'hidden_layer_sizes': (100, 50),
                'max_iter': 500,
            },
            'gradient_boosting': {
                'random_state': self.random_state,
                'n_estimators': 100,
            },
            'decision_tree': {
                'random_state': self.random_state,
                'max_depth': 10,
            },
        }
        return defaults.get(self.algorithm, {})
    
    def evaluate_model(self):
        """Evaluate the trained model and return metrics"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred) * 100,
            'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
        }
        
        # Get confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        return metrics, cm, y_pred
    
    def save_model(self, output_path):
        """Save the trained model and preprocessing objects"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'algorithm': self.algorithm,
        }
        
        joblib.dump(model_data, output_path)
        return output_path
    
    def generate_confusion_matrix_plot(self):
        """Generate confusion matrix visualization"""
        _, cm, _ = self.evaluate_model()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(self.y_test),
                    yticklabels=np.unique(self.y_test))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plt.close()
        
        return ContentFile(buffer.read(), name='confusion_matrix.png')
    
    def generate_feature_importance_plot(self):
        """Generate feature importance plot (for tree-based models)"""
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10 features
        
        plt.figure(figsize=(10, 6))
        plt.title('Top 10 Feature Importances')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), 
                   [self.feature_names[i] for i in indices], 
                   rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        # Save to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plt.close()
        
        return ContentFile(buffer.read(), name='feature_importance.png')


def load_model(model_path):
    """Load a saved model"""
    return joblib.load(model_path)


def predict(model_path, input_data):
    """Make predictions using a saved model"""
    model_data = load_model(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Preprocess input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    predictions = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled) if hasattr(model, 'predict_proba') else None
    
    return predictions, probabilities