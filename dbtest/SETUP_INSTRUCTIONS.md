# Model Testing System Setup Instructions

## Step 1: Install Required Packages

```bash
pip install pytest pytest-django pytest-cov factory-boy scikit-learn joblib
```

## Step 2: Project Structure

Create the following directory structure in your `dbtest` app:

```
dbtest/
├── __init__.py
├── admin.py              # (provided)
├── models.py             # (provided)
├── views.py              # (provided)
├── urls.py               # (provided)
├── test_runner.py        # (provided)
├── fixtures/
│   ├── __init__.py       # (create empty file)
│   └── datasets.py       # (provided)
├── templates/
│   └── dbtest/
│       └── dashboard.html  # (provided)
└── tests/
    └── __init__.py       # (create empty file)
```

## Step 3: Update Your Main URLs

In your project's main `urls.py`, add the dbtest URLs:

```python
# your_project/urls.py
from django.urls import path, include

urlpatterns = [
    # ... your existing URLs ...
    path('tests/', include('dbtest.urls')),
]
```

## Step 4: Update Settings

Make sure these apps are in your `INSTALLED_APPS`:

```python
# settings.py
INSTALLED_APPS = [
    # ... other apps ...
    'dbtest',
    'dashboard',
    'core',
]
```

## Step 5: Create Migrations and Migrate

```bash
python manage.py makemigrations dbtest
python manage.py migrate
```

## Step 6: Add Navigation Link

Add a link to the test dashboard in your base template navigation:

```html
<!-- In your base.html or navigation template -->
<a href="{% url 'test_dashboard' %}" class="nav-link">
    <i class="bi bi-shield-check"></i> Model Tests
</a>
```

## Step 7: Create pytest.ini

Create `pytest.ini` in your project root (replace `your_project` with your actual project name):

```ini
[pytest]
DJANGO_SETTINGS_MODULE = your_project.settings
python_files = tests.py test_*.py *_tests.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    slow: marks tests as slow
    fast: marks tests as fast
    classification: classification model tests
    regression: regression model tests
testpaths = dbtest/tests
```

## Step 8: Verify Installation

Run the Django development server:

```bash
python manage.py runserver
```

Navigate to: `http://localhost:8000/tests/`

## Step 9: Run Your First Test

1. Click "Run All Tests" button
2. Wait for tests to complete (may take 2-5 minutes)
3. View results on the dashboard

## File Checklist

Ensure you have created/updated these files:

- ✓ `dbtest/models.py` - TestRun and TestResult models
- ✓ `dbtest/views.py` - Test views
- ✓ `dbtest/urls.py` - URL routing
- ✓ `dbtest/admin.py` - Admin configuration
- ✓ `dbtest/test_runner.py` - Core test runner logic
- ✓ `dbtest/fixtures/__init__.py` - Empty file
- ✓ `dbtest/fixtures/datasets.py` - Dataset generators
- ✓ `dbtest/templates/dbtest/dashboard.html` - Main template
- ✓ `dbtest/tests/__init__.py` - Empty file
- ✓ `pytest.ini` - Pytest configuration (project root)
- ✓ Main `urls.py` - Added dbtest URLs
- ✓ `settings.py` - Added dbtest to INSTALLED_APPS

## Troubleshooting

### Issue: Models not found
**Solution**: Run migrations:
```bash
python manage.py makemigrations dbtest
python manage.py migrate
```

### Issue: Templates not found
**Solution**: Ensure template directory exists: `dbtest/templates/dbtest/`

### Issue: Import errors
**Solution**: Install required packages:
```bash
pip install scikit-learn joblib pandas numpy
```

### Issue: Permission errors with test_models directory
**Solution**: Ensure MEDIA_ROOT is set in settings.py and is writable

### Issue: Tests timing out
**Solution**: This is normal for first run. Neural networks and SVMs can be slow.

## Usage

### Run All Tests
Click "Run All Tests" to test all 24 models (12 classification + 12 regression)

### Run Specific Type
- Click "Classification" to test only classification models
- Click "Regression" to test only regression models

### View Results
Results are displayed immediately on the same page after tests complete

### Cleanup
Click "Cleanup" to delete all test runs and free up disk space

## What Gets Tested

For each model:
1. ✓ Model initialization with default parameters
2. ✓ Training on synthetic dataset
3. ✓ Making predictions
4. ✓ Calculating metrics (accuracy/R²)
5. ✓ Saving model to disk
6. ✓ Loading model from disk
7. ✓ Predictions match after reload

## Expected Results

All models should pass except:
- Neural networks may occasionally fail with convergence warnings
- SVMs may be slow on large datasets
- Some models may require specific data conditions

Typical run time: 2-5 minutes for all models

## Next Steps

Once basic testing works:
1. Add more test scenarios (missing values, edge cases)
2. Implement async testing with Celery for better UX
3. Add performance benchmarking
4. Create comparison visualizations
5. Set up CI/CD integration