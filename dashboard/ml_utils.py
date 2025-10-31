"""
Machine Learning training utilities for scikit-learn models
Implements all required model families with clean API
"""
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score, mean_absolute_error, roc_curve, auc
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from django.core.files.base import ContentFile


class DataValidator:
    """Validates and checks dataset schema and quality"""
    
    @staticmethod
    def validate_file_size(file_size, max_size_mb=50):
        """Check if file size is within limits"""
        max_bytes = max_size_mb * 1024 * 1024
        if file_size > max_bytes:
            raise ValueError(f"File size exceeds {max_size_mb}MB limit")
        return True
    
    @staticmethod
    def validate_schema(df):
        """Validate dataset schema and structure"""
        issues = []
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if len(df) < 10:
            issues.append(f"Dataset has only {len(df)} rows. Minimum recommended: 10")
        
        if df.columns.duplicated().any():
            duplicates = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate column names found: {duplicates}")
        
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            issues.append(f"Columns with all null values: {null_cols}")
        
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95:
                issues.append(f"Column '{col}' has high cardinality ({unique_ratio:.2%}). May be an ID column.")
        
        type_info = {
            'numeric': len(df.select_dtypes(include=[np.number]).columns),
            'categorical': len(df.select_dtypes(include=['object']).columns),
            'datetime': len(df.select_dtypes(include=['datetime']).columns),
        }
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'type_info': type_info,
            'shape': df.shape,
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        }
    
    @staticmethod
    def check_target_column(df, target_column):
        """Validate target column suitability"""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        target = df[target_column]
        
        if target.isnull().any():
            raise ValueError(f"Target column '{target_column}' contains null values")
        
        n_unique = target.nunique()
        if n_unique == 1:
            raise ValueError(f"Target column '{target_column}' has only one unique value")
        
        # Determine if classification or regression
        task_type = 'classification' if n_unique < 20 or target.dtype == 'object' else 'regression'
        
        return {
            'valid': True,
            'task_type': task_type,
            'n_unique': n_unique,
            'class_distribution': target.value_counts().to_dict() if task_type == 'classification' else None
        }


class PreprocessingConfig:
    """Store and manage preprocessing configuration"""
    
    def __init__(self):
        self.config = {
            'missing_value_strategy': 'mean',
            'encoding_strategy': 'label',
            'scaling_strategy': 'standard',
            'feature_columns': [],
            'target_column': None,
            'categorical_columns': [],
            'numeric_columns': [],
            'column_dtypes': {},
            'train_test_split': 0.2,
            'validation_split': 0.0,
            'random_state': 42,
        }
    
    def update(self, **kwargs):
        """Update configuration"""
        self.config.update(kwargs)
    
    def save(self, path):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def load(cls, path):
        """Load configuration from JSON"""
        config = cls()
        with open(path, 'r') as f:
            config.config = json.load(f)
        return config
    
    def to_dict(self):
        """Return configuration as dictionary"""
        return self.config.copy()


class ModelRegistry:
    """
    Central registry for all ML models with metadata
    Clean API to add new models
    """
    
    # Classification Models
    CLASSIFICATION_MODELS = {
        # Linear Models
        'logistic_regression': {
            'class': LogisticRegression,
            'name': 'Logistic Regression',
            'family': 'linear',
            'description': 'Linear model for classification',
            'default_params': {
                'random_state': 42,
                'max_iter': 1000,
            },
            'tunable_params': ['C', 'penalty', 'solver']
        },
        
        # Tree-based Models
        'decision_tree': {
            'class': DecisionTreeClassifier,
            'name': 'Decision Tree',
            'family': 'tree',
            'description': 'Single decision tree classifier',
            'default_params': {
                'random_state': 42,
                'max_depth': 10,
            },
            'tunable_params': ['max_depth', 'min_samples_split', 'min_samples_leaf']
        },
        
        # Ensemble - Bagging
        'random_forest': {
            'class': RandomForestClassifier,
            'name': 'Random Forest',
            'family': 'bagging',
            'description': 'Ensemble of decision trees',
            'default_params': {
                'n_estimators': 100,
                'random_state': 42,
                'max_depth': 10,
            },
            'tunable_params': ['n_estimators', 'max_depth', 'min_samples_split']
        },
        'bagging': {
            'class': BaggingClassifier,
            'name': 'Bagging Classifier',
            'family': 'bagging',
            'description': 'Bootstrap aggregating classifier',
            'default_params': {
                'n_estimators': 50,
                'random_state': 42,
            },
            'tunable_params': ['n_estimators', 'max_samples', 'max_features']
        },
        
        # Ensemble - Boosting
        'gradient_boosting': {
            'class': GradientBoostingClassifier,
            'name': 'Gradient Boosting',
            'family': 'boosting',
            'description': 'Gradient boosted decision trees',
            'default_params': {
                'n_estimators': 100,
                'random_state': 42,
                'learning_rate': 0.1,
            },
            'tunable_params': ['n_estimators', 'learning_rate', 'max_depth']
        },
        'adaboost': {
            'class': AdaBoostClassifier,
            'name': 'AdaBoost',
            'family': 'boosting',
            'description': 'Adaptive boosting classifier',
            'default_params': {
                'n_estimators': 50,
                'random_state': 42,
                'learning_rate': 1.0,
            },
            'tunable_params': ['n_estimators', 'learning_rate']
        },
        
        # Support Vector Machines
        'svm': {
            'class': SVC,
            'name': 'Support Vector Machine',
            'family': 'svm',
            'description': 'Support vector classifier',
            'default_params': {
                'random_state': 42,
                'kernel': 'rbf',
            },
            'tunable_params': ['C', 'kernel', 'gamma']
        },
        
        # Neural Networks
        'neural_network': {
            'class': MLPClassifier,
            'name': 'Neural Network (MLP)',
            'family': 'neural_network',
            'description': 'Multi-layer perceptron classifier',
            'default_params': {
                'random_state': 42,
                'hidden_layer_sizes': (100, 50),
                'max_iter': 500,
                'early_stopping': True,
            },
            'tunable_params': ['hidden_layer_sizes', 'learning_rate_init', 'alpha']
        },
    }
    
    # Regression Models
    REGRESSION_MODELS = {
        # Linear Models
        'linear_regression': {
            'class': LinearRegression,
            'name': 'Linear Regression',
            'family': 'linear',
            'description': 'Ordinary least squares regression',
            'default_params': {},
            'tunable_params': ['fit_intercept', 'normalize']
        },
        'ridge': {
            'class': Ridge,
            'name': 'Ridge Regression',
            'family': 'linear',
            'description': 'Linear regression with L2 regularization',
            'default_params': {
                'random_state': 42,
                'alpha': 1.0,
            },
            'tunable_params': ['alpha', 'solver']
        },
        'lasso': {
            'class': Lasso,
            'name': 'Lasso Regression',
            'family': 'linear',
            'description': 'Linear regression with L1 regularization',
            'default_params': {
                'random_state': 42,
                'alpha': 1.0,
            },
            'tunable_params': ['alpha', 'max_iter']
        },
        
        # Tree-based Models
        'decision_tree': {
            'class': DecisionTreeRegressor,
            'name': 'Decision Tree',
            'family': 'tree',
            'description': 'Single decision tree regressor',
            'default_params': {
                'random_state': 42,
                'max_depth': 10,
            },
            'tunable_params': ['max_depth', 'min_samples_split', 'min_samples_leaf']
        },
        
        # Ensemble - Bagging
        'random_forest': {
            'class': RandomForestRegressor,
            'name': 'Random Forest',
            'family': 'bagging',
            'description': 'Ensemble of decision trees',
            'default_params': {
                'n_estimators': 100,
                'random_state': 42,
                'max_depth': 10,
            },
            'tunable_params': ['n_estimators', 'max_depth', 'min_samples_split']
        },
        'bagging': {
            'class': BaggingRegressor,
            'name': 'Bagging Regressor',
            'family': 'bagging',
            'description': 'Bootstrap aggregating regressor',
            'default_params': {
                'n_estimators': 50,
                'random_state': 42,
            },
            'tunable_params': ['n_estimators', 'max_samples', 'max_features']
        },
        
        # Ensemble - Boosting
        'gradient_boosting': {
            'class': GradientBoostingRegressor,
            'name': 'Gradient Boosting',
            'family': 'boosting',
            'description': 'Gradient boosted decision trees',
            'default_params': {
                'n_estimators': 100,
                'random_state': 42,
                'learning_rate': 0.1,
            },
            'tunable_params': ['n_estimators', 'learning_rate', 'max_depth']
        },
        'adaboost': {
            'class': AdaBoostRegressor,
            'name': 'AdaBoost',
            'family': 'boosting',
            'description': 'Adaptive boosting regressor',
            'default_params': {
                'n_estimators': 50,
                'random_state': 42,
                'learning_rate': 1.0,
            },
            'tunable_params': ['n_estimators', 'learning_rate']
        },
        
        # Support Vector Machines
        'svm': {
            'class': SVR,
            'name': 'Support Vector Machine',
            'family': 'svm',
            'description': 'Support vector regressor',
            'default_params': {
                'kernel': 'rbf',
            },
            'tunable_params': ['C', 'kernel', 'gamma', 'epsilon']
        },
        
        # Neural Networks
        'neural_network': {
            'class': MLPRegressor,
            'name': 'Neural Network (MLP)',
            'family': 'neural_network',
            'description': 'Multi-layer perceptron regressor',
            'default_params': {
                'random_state': 42,
                'hidden_layer_sizes': (100, 50),
                'max_iter': 500,
                'early_stopping': True,
            },
            'tunable_params': ['hidden_layer_sizes', 'learning_rate_init', 'alpha']
        },
    }
    
    @classmethod
    def get_model(cls, algorithm, task_type='classification'):
        """Get model class and metadata"""
        models = cls.CLASSIFICATION_MODELS if task_type == 'classification' else cls.REGRESSION_MODELS
        return models.get(algorithm)
    
    @classmethod
    def list_models(cls, task_type='classification', family=None):
        """List available models, optionally filtered by family"""
        models = cls.CLASSIFICATION_MODELS if task_type == 'classification' else cls.REGRESSION_MODELS
        if family:
            return {k: v for k, v in models.items() if v['family'] == family}
        return models
    
    @classmethod
    def get_families(cls):
        """Get list of model families"""
        return ['linear', 'tree', 'bagging', 'boosting', 'svm', 'neural_network']


class ModelTrainer:
    """Handles training and evaluation of ML models with consistent API"""
    
    def __init__(self, dataset_path, target_column, algorithm='random_forest', 
                 task_type='classification', test_size=0.2, validation_size=0.0,
                 random_state=42, missing_value_strategy='mean', algorithm_params=None):
        """
        Initialize the trainer with consistent input contract
        
        Args:
            dataset_path: Path to dataset file
            target_column: Name of target variable
            algorithm: Algorithm identifier
            task_type: 'classification' or 'regression'
            test_size: Test set proportion
            validation_size: Validation set proportion
            random_state: Random seed
            missing_value_strategy: Strategy for missing values
            algorithm_params: Custom hyperparameters
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.algorithm = algorithm
        self.task_type = task_type
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.missing_value_strategy = missing_value_strategy
        self.algorithm_params = algorithm_params or {}
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.feature_names = None
        self.scaler = None
        self.label_encoders = {}
        self.preprocessing_config = PreprocessingConfig()
        self.validator = DataValidator()
        # Detected task_type after inspecting the target column (set in load_and_preprocess_data)
        self.detected_task_type = None
        
    def load_and_preprocess_data(self):
        """Load dataset and perform preprocessing"""
        ext = os.path.splitext(self.dataset_path)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(self.dataset_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(self.dataset_path)
        elif ext == '.json':
            df = pd.read_json(self.dataset_path)
        elif ext == '.txt':
            try:
                df = pd.read_csv(self.dataset_path, sep='\t')
            except:
                try:
                    df = pd.read_csv(self.dataset_path, sep=',')
                except:
                    df = pd.read_csv(self.dataset_path, delim_whitespace=True)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Validate schema
        schema_validation = self.validator.validate_schema(df)
        if not schema_validation['valid']:
            print(f"WARNING: Schema validation issues: {schema_validation['issues']}")
        
        # Validate target
        target_validation = self.validator.check_target_column(df, self.target_column)
        print(f"Task type detected: {target_validation['task_type']}")
        # store detected task type so trainer can pick the correct model family
        n_unique = df[self.target_column].nunique()
        if pd.api.types.is_numeric_dtype(df[self.target_column]) and n_unique > 10:
            self.detected_task_type = 'regression'
        else:
            self.detected_task_type = 'classification'

        
        # Store original dtypes
        # If the caller did not explicitly set a task_type, prefer the detected one
        # Keep original self.task_type so callers that force a type still have that value
        self.preprocessing_config.update(
            column_dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            target_column=self.target_column,
            task_type=self.detected_task_type or self.task_type
        )
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        # Always ensure it's a DataFrame
        if isinstance(X, pd.Series):
            X = X.to_frame()


        y = df[self.target_column].values

        # Identify column types
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        self.preprocessing_config.update(
            categorical_columns=categorical_cols,
            numeric_columns=numeric_cols,
            feature_columns=X.columns.tolist()
        )
        
        # Handle missing values
        X = self._handle_missing_values(X, numeric_cols, categorical_cols)
        
        # Handle categorical features
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders['target'] = le
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        if self.validation_size > 0:
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            val_size_adjusted = self.validation_size / (1 - self.test_size)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        if self.validation_size > 0:
            self.X_val = self.scaler.transform(self.X_val)
        
        # Update config
        self.preprocessing_config.update(
            missing_value_strategy=self.missing_value_strategy,
            train_test_split=self.test_size,
            validation_split=self.validation_size,
            random_state=self.random_state
        )
        
        return True
    
    def _handle_missing_values(self, X, numeric_cols, categorical_cols):
        """Handle missing values based on strategy"""
        if self.missing_value_strategy == 'drop':
            X = X.dropna()
        elif self.missing_value_strategy == 'mean':
            for col in numeric_cols:
                X[col] = X[col].fillna(X[col].mean())
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        elif self.missing_value_strategy == 'median':
            for col in numeric_cols:
                X[col] = X[col].fillna(X[col].median())
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        elif self.missing_value_strategy == 'mode':
            for col in X.columns:
                if not X[col].mode().empty:
                    X[col] = X[col].fillna(X[col].mode()[0])
        
        return X
    
    def train_model(self):
        """Train the selected model with consistent output contract"""
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first")
        
        # Decide effective task type: prefer detected if available, otherwise use configured
        effective_task = self.detected_task_type or self.task_type

        # Get model from registry
        model_info = ModelRegistry.get_model(self.algorithm, effective_task)
        if not model_info:
            raise ValueError(f"Unknown algorithm: {self.algorithm} for task: {effective_task}")
        
        # Merge default params with custom params
        params = {**model_info['default_params'], **self.algorithm_params}
        
        # Handle random_state for models that don't support it
        if 'random_state' in params and model_info['class'] == LinearRegression:
            params.pop('random_state')
        
        # Initialize and train model
        self.model = model_info['class'](**params)
        self.model.fit(self.X_train, self.y_train)
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate with consistent metrics for both classification and regression"""
        if self.model is None:
            raise ValueError("Model not trained")

        # use detected task if available
        effective_task = self.detected_task_type or self.task_type

        y_pred = self.model.predict(self.X_test)

        if effective_task == 'classification':
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred) * 100,
                'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
            }
            cm = confusion_matrix(self.y_test, y_pred)
        else:  # regression
            metrics = {
                'mse': mean_squared_error(self.y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                'mae': mean_absolute_error(self.y_test, y_pred),
                'r2_score': r2_score(self.y_test, y_pred),
            }
            cm = None
        
        return metrics, cm, y_pred
    
    def save_model(self, output_path):
        """Save model with consistent contract"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        effective_task = self.task_type or self.detected_task_type

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'algorithm': self.algorithm,
            'task_type': effective_task,
            'preprocessing_config': self.preprocessing_config.to_dict(),
        }
        
        joblib.dump(model_data, output_path)
        
        # Save config as JSON
        config_path = output_path.replace('.joblib', '_config.json')
        self.preprocessing_config.save(config_path)
        
        return output_path
    
    def generate_confusion_matrix_plot(self, max_classes=10):
        """Generate simplified confusion matrix for classification"""
        effective_task = self.detected_task_type or self.task_type
        if effective_task != 'classification':
            return None
        
        _, cm, y_test = self.evaluate_model()

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

    
    def generate_roc_curve_plot(self):
        """Generate ROC curve for classification models with probability output"""
        effective_task = self.task_type
        if effective_task != 'classification':
            return None

        # Ensure model supports probability estimates
        if not hasattr(self.model, "predict_proba"):
            print("Model does not support probability estimates for ROC curve.")
            return None

        y_prob = self.model.predict_proba(self.X_test)
        y_true = self.y_test

        plt.figure(figsize=(8, 6))
        if y_prob.shape[1] == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # Multiclass — average ROC
            max_classes_to_plot = 10
            for i in range(min(y_prob.shape[1], max_classes_to_plot)):
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, weight='bold')
        plt.legend(loc="lower right", fontsize=8, ncol=2)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120)
        buffer.seek(0)
        plt.close()

        return ContentFile(buffer.read(), name='roc_curve.png')
    
    def generate_feature_importance_plot(self):
        """Generate feature importance plot"""
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        plt.figure(figsize=(10, 6))
        plt.title('Top 10 Feature Importances')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), 
                   [self.feature_names[i] for i in indices], 
                   rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plt.close()
        
        return ContentFile(buffer.read(), name='feature_importance.png')


def load_model(model_path):
    """Load a saved model"""
    return joblib.load(model_path)


def predict(model_path, input_data):
    """Make predictions with consistent API"""
    model_data = load_model(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    
    input_scaled = scaler.transform(input_data)
    
    predictions = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled) if hasattr(model, 'predict_proba') else None
    
    return predictions, probabilities