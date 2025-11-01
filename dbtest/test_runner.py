"""
Test runner for model testing functionality.
Handles running tests on ML models and storing results.
"""
from django.utils import timezone
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import os
import shutil

from .models import TestRun, TestResult
from dashboard.ml_utils import ModelRegistry


class ModelTestRunner:
    """Runs tests on ML models and records results"""
    
    TEST_DATA_DIR = 'test_models'
    
    def __init__(self, test_type='all', user=None):
        """Initialize test runner with type of tests to run"""
        self.test_type = test_type
        self.user = user
        self.test_run = None
    
    def run_all_tests(self):
        """Run all tests based on test_type"""
        # Create test run record
        self.test_run = TestRun.objects.create(
            test_type=self.test_type,
            status='running'
        )
        
        try:
            # Build a list of (algo_name, model_meta, task_type) entries so we
            # test both classification and regression variants separately.
            to_test = []
            if self.test_type in ('classification', 'all'):
                for algo_name, model_meta in ModelRegistry.CLASSIFICATION_MODELS.items():
                    to_test.append((algo_name, model_meta, 'classification'))
            if self.test_type in ('regression', 'all'):
                for algo_name, model_meta in ModelRegistry.REGRESSION_MODELS.items():
                    to_test.append((algo_name, model_meta, 'regression'))

            # Run tests for each model variant
            for algo_name, model_meta, task in to_test:
                self._test_model(algo_name, model_meta, task)
            
            # Update test run status
            self.test_run.status = 'completed'
            
        except Exception as e:
            self.test_run.status = 'failed'
            # Create error result
            TestResult.objects.create(
                test_run=self.test_run,
                test_case='TestRunner',
                test_method='run_all_tests',
                status='error',
                error_message=str(e),
                error_traceback='Error during test execution'
            )
        
        finally:
            self.test_run.save()
        
        return self.test_run
    
    def _test_model(self, algo_name, model_meta, task_type='classification'):
        """Test a specific model variant (task_type distinguishes classification/regression)"""
        model_class = model_meta['class']
        
        try:
            # Generate synthetic data
            X, y = self._generate_test_data(task_type)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize and train model
            model = model_class(**model_meta.get('default_params', {}))
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics and record them in TestResult.metrics
            metrics = {}
            primary_metric = None
            if task_type == 'classification':
                accuracy = float((y_pred == y_test).mean())
                metrics['accuracy'] = accuracy
                primary_metric = accuracy
                result_status = 'pass' if accuracy >= 0.6 else 'fail'
                message = f'Accuracy: {accuracy:.2f}'
            else:
                r2 = float(model.score(X_test, y_test))
                metrics['r2_score'] = r2
                primary_metric = r2
                result_status = 'pass' if r2 >= 0.5 else 'fail'
                message = f'R² Score: {r2:.2f}'

            # Record result with richer metadata expected by templates
            TestResult.objects.create(
                test_run=self.test_run,
                test_case=model_meta.get('name', algo_name),
                test_method='basic_performance',
                status=result_status,
                model_name=model_meta.get('name', algo_name),
                algorithm=algo_name,
                task_type=task_type,
                metrics=metrics,
                primary_metric=primary_metric,
                error_message=message if result_status == 'fail' else None,
                duration=0.0,
                train_time=0.0
            )

            if result_status == 'pass':
                self.test_run.passed += 1
            else:
                self.test_run.failed += 1
                
        except Exception as e:
            # Record error
            TestResult.objects.create(
                test_run=self.test_run,
                test_case=algo_name,
                test_method='basic_performance',
                status='error',
                error_message=str(e),
                error_traceback='Error during model testing'
            )
            self.test_run.errors += 1
        
        self.test_run.save()
    
    def _generate_test_data(self, task_type, n_samples=1000):
        """Generate synthetic data based on task type"""
        if task_type == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                random_state=42
            )
        else:
            X, y = make_regression(
                n_samples=n_samples,
                n_features=20,
                n_informative=15,
                noise=0.1,
                random_state=42
            )
        return X, y
    
    @classmethod
    def cleanup_test_models(cls):
        """Clean up any test model files"""
        if os.path.exists(cls.TEST_DATA_DIR):
            shutil.rmtree(cls.TEST_DATA_DIR)