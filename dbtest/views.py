from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from .models import TestRun, TestResult
from .test_runner import ModelTestRunner
from dashboard.ml_utils import ModelRegistry


@login_required
def test_dashboard(request):
    """Main test dashboard - shows all test results in one page"""
    # Get latest test run
    latest_run = TestRun.objects.first()
    
    # Get all test runs for history
    test_runs = TestRun.objects.all()[:10]  # Last 10 runs
    
    # Get classification and regression model counts
    classification_count = len(ModelRegistry.CLASSIFICATION_MODELS)
    regression_count = len(ModelRegistry.REGRESSION_MODELS)
    
    latest_failed_pct = 0
    if latest_run and latest_run.total_tests > 0:
        latest_failed_pct = (latest_run.failed / latest_run.total_tests) * 100.0

    context = {
        'latest_run': latest_run,
        'latest_failed_pct': latest_failed_pct,
        'test_runs': test_runs,
        'classification_count': classification_count,
        'regression_count': regression_count,
        'total_models': classification_count + regression_count,
    }
    
    return render(request, 'dbtest/dashboard.html', context)


@login_required
@require_http_methods(["POST"])
def run_tests(request):
    """Run tests based on test_type parameter"""
    test_type = request.POST.get('test_type', 'all')
    
    if test_type not in ['all', 'classification', 'regression']:
        messages.error(request, 'Invalid test type')
        return redirect('test_dashboard')
    
    try:
        # Run tests
        runner = ModelTestRunner(test_type=test_type, user=request.user)
        test_run = runner.run_all_tests()
        
        # Cleanup test models after run
        ModelTestRunner.cleanup_test_models()
        
        messages.success(
            request, 
            f'Tests completed! {test_run.passed}/{test_run.total_tests} passed'
        )
        
    except Exception as e:
        messages.error(request, f'Error running tests: {str(e)}')
    
    return redirect('test_dashboard')


@login_required
def test_results(request, run_id):
    """View detailed results for a specific test run"""
    try:
        test_run = TestRun.objects.get(id=run_id)
        results = test_run.results.all()
        
        # Separate by task type
        classification_results = results.filter(task_type='classification')
        regression_results = results.filter(task_type='regression')
        
        context = {
            'test_run': test_run,
            'classification_results': classification_results,
            'regression_results': regression_results,
        }
        
        return render(request, 'dbtest/results.html', context)
        
    except TestRun.DoesNotExist:
        messages.error(request, 'Test run not found')
        return redirect('test_dashboard')


@login_required
def cleanup_tests(request):
    """Delete all test runs and cleanup test models"""
    if request.method == 'POST':
        # Delete all test runs (cascade deletes results)
        count = TestRun.objects.count()
        TestRun.objects.all().delete()
        
        # Cleanup test model files
        ModelTestRunner.cleanup_test_models()
        
        messages.success(request, f'Cleaned up {count} test runs')
    
    return redirect('test_dashboard')


@login_required
def get_test_status(request, run_id):
    """API endpoint to get current test status (for AJAX polling if needed)"""
    try:
        test_run = TestRun.objects.get(id=run_id)
        data = {
            'status': test_run.status,
            'total_tests': test_run.total_tests,
            'passed': test_run.passed,
            'failed': test_run.failed,
            'progress': (test_run.passed + test_run.failed) / test_run.total_tests * 100 if test_run.total_tests > 0 else 0,
        }
        return JsonResponse(data)
    except TestRun.DoesNotExist:
        return JsonResponse({'error': 'Test run not found'}, status=404)