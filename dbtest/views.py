from django.shortcuts import render, redirect
from .forms import UploadedFileForm, AIModelForm, PerformanceMetricForm, PredictionForm, FigureForm
from core.models import UploadedFile, AIModel, PerformanceMetric, Prediction, Figure

def database_test(request):
    forms = {
        'uploaded_file_form': UploadedFileForm(prefix='file'),
        'ai_model_form': AIModelForm(prefix='model'),
        'metric_form': PerformanceMetricForm(prefix='metric'),
        'prediction_form': PredictionForm(prefix='pred'),
        'figure_form': FigureForm(prefix='fig'),
    }

    if request.method == 'POST':
        if 'submit_file' in request.POST:
            form = UploadedFileForm(request.POST, request.FILES, prefix='file')
            if form.is_valid():
                form.save()
        elif 'submit_model' in request.POST:
            form = AIModelForm(request.POST, prefix='model')
            if form.is_valid():
                form.save()
        elif 'submit_metric' in request.POST:
            form = PerformanceMetricForm(request.POST, prefix='metric')
            if form.is_valid():
                form.save()
        elif 'submit_pred' in request.POST:
            form = PredictionForm(request.POST, prefix='pred')
            if form.is_valid():
                form.save()
        elif 'submit_fig' in request.POST:
            form = FigureForm(request.POST, request.FILES, prefix='fig')
            if form.is_valid():
                form.save()
        return redirect('dbtest')

    context = {
        **forms,
        'files': UploadedFile.objects.all(),
        'models': AIModel.objects.all(),
        'metrics': PerformanceMetric.objects.all(),
        'predictions': Prediction.objects.all(),
        'figures': Figure.objects.all(),
    }
    return render(request, 'dbtest/test_dashboard.html', context)