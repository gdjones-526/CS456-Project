from django.db import models
from django.utils import timezone


class TestRun(models.Model):
    TEST_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    TEST_TYPE_CHOICES = [
        ('unit', 'Unit Tests'),
        ('integration', 'Integration Tests'),
        ('all', 'All Tests'),
    ]
    
    timestamp = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=20, choices=TEST_STATUS_CHOICES, default='pending')
    test_type = models.CharField(max_length=20, choices=TEST_TYPE_CHOICES, default='all')
    passed = models.IntegerField(default=0)
    failed = models.IntegerField(default=0)
    errors = models.IntegerField(default=0)
    skipped = models.IntegerField(default=0)
    duration = models.FloatField(default=0.0)  # Duration in seconds
    
    @property
    def total_tests(self):
        """Total number of tests run"""
        return self.passed + self.failed + self.errors + self.skipped
    
    @property
    def pass_rate(self):
        """Pass rate as a percentage (0-100).

        Templates expect a numeric percentage value (e.g. 81.8) so return
        the value multiplied by 100. This avoids doing math in templates.
        """
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100.0
    
    @property
    def pass_rate_display(self):
        """Pass rate as a percentage string"""
        return f"{self.pass_rate:.1f}%"
    
    def __str__(self):
        return f"Test Run {self.id} ({self.test_type}) - {self.status}"
    
    class Meta:
        ordering = ['-timestamp']


class TestResult(models.Model):
    STATUS_CHOICES = [
        ('pass', 'Pass'),
        ('fail', 'Fail'),
        ('error', 'Error'),
        ('skipped', 'Skipped'),
    ]
    
    test_run = models.ForeignKey(TestRun, on_delete=models.CASCADE, related_name='results')
    test_case = models.CharField(max_length=255)  # Name of the test case
    test_method = models.CharField(max_length=255)  # Name of the specific test method
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    # Additional metadata for UI and reporting
    model_name = models.CharField(max_length=255, blank=True, null=True)
    algorithm = models.CharField(max_length=255, blank=True, null=True)
    task_type = models.CharField(max_length=50, blank=True, null=True)
    metrics = models.JSONField(blank=True, null=True)
    train_time = models.FloatField(default=0.0)
    predict_time = models.FloatField(default=0.0)
    dataset_size = models.IntegerField(blank=True, null=True)
    model_path = models.CharField(max_length=1024, blank=True, null=True)
    primary_metric = models.FloatField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)
    error_traceback = models.TextField(blank=True, null=True)
    duration = models.FloatField(default=0.0)  # Duration in seconds
    timestamp = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.test_case}.{self.test_method} - {self.status}"
    
    class Meta:
        ordering = ['test_case', 'test_method']
