from django.db import models
from django.contrib.auth.models import User


class UploadedFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_files')
    file = models.FileField(upload_to='datasets/')
    original_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.original_name} (by {self.user.username})"


class AIModel(models.Model):
    STATUS_CHOICES = [
        ('training', 'Training'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='models')
    dataset = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, related_name='models')
    name = models.CharField(max_length=100)
    framework = models.CharField(max_length=100, default='scikit-learn')
    target_variable = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    model_file = models.FileField(upload_to='trained_models/', blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='training')
    parameters = models.JSONField(blank=True, null=True)

    def __str__(self):
        return f"{self.name} ({self.user.username})"

class PerformanceMetric(models.Model):
    model = models.ForeignKey('AIModel', on_delete=models.CASCADE, related_name='metrics')
    
    # Classification metrics
    accuracy = models.FloatField(blank=True, null=True)
    precision = models.FloatField(blank=True, null=True)
    recall = models.FloatField(blank=True, null=True)
    f1_score = models.FloatField(blank=True, null=True)
    
    loss = models.FloatField(blank=True, null=True)

    # Regression metrics
    mse = models.FloatField(blank=True, null=True)
    rmse = models.FloatField(blank=True, null=True)
    mae = models.FloatField(blank=True, null=True)
    r2_score = models.FloatField(blank=True, null=True)
    
    # Common
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Metrics for {self.model.name}"
    
class Prediction(models.Model):
    model = models.ForeignKey(AIModel, on_delete=models.CASCADE, related_name='predictions')
    input_data = models.JSONField()
    predicted_output = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction {self.id} for {self.model.name}"


class Figure(models.Model):
    model = models.ForeignKey(AIModel, on_delete=models.CASCADE, related_name='figures')
    figure_file = models.ImageField(upload_to='figures/')
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Figure for {self.model.name}"


class Experiment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='experiments')
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Experiment: {self.name}"