from django import forms
from core.models import UploadedFile, AIModel, PerformanceMetric, Prediction, Figure

class UploadedFileForm(forms.ModelForm):
    class Meta:
        model = UploadedFile
        fields = ['user', 'file', 'original_name', 'description']

class AIModelForm(forms.ModelForm):
    class Meta:
        model = AIModel
        fields = ['user', 'dataset', 'name', 'framework', 'target_variable', 'parameters']

class PerformanceMetricForm(forms.ModelForm):
    class Meta:
        model = PerformanceMetric
        fields = ['model', 'accuracy', 'precision', 'recall', 'f1_score', 'loss']

class PredictionForm(forms.ModelForm):
    class Meta:
        model = Prediction
        fields = ['model', 'input_data', 'predicted_output']

class FigureForm(forms.ModelForm):
    class Meta:
        model = Figure
        fields = ['model', 'figure_file', 'description']