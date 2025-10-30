from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from core.models import UploadedFile, AIModel, PerformanceMetric, Figure
from .forms import FileUploadForm, ModelTrainingForm
from .ml_utils import *
import pandas as pd
import os
import json
from django.conf import settings
from django.http import JsonResponse
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout, authenticate

@login_required
def upload_file(request):
    """Dedicated file upload page"""
    recent_files = UploadedFile.objects.filter(user=request.user).order_by('-uploaded_at')[:5]
    
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save(commit=False)
            uploaded_file.user = request.user
            uploaded_file.original_name = request.FILES['file'].name
            uploaded_file.save()
            
            # Process the file to validate it
            try:
                process_uploaded_file(uploaded_file)
                messages.success(request, f'File "{uploaded_file.original_name}" uploaded successfully!')
                return redirect('file_detail', pk=uploaded_file.pk)
            except Exception as e:
                uploaded_file.delete()
                messages.error(request, f'Error processing file: {str(e)}')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = FileUploadForm()
    
    context = {
        'form': form,
        'recent_files': recent_files,
    }
    return render(request, 'upload_file.html', context)


@login_required
def dashboard(request):
    """Main dashboard view with file upload and management"""
    files = UploadedFile.objects.filter(user=request.user).order_by('-uploaded_at')
    trained_models = AIModel.objects.filter(user=request.user).prefetch_related('metrics')
    
    # Get latest metric for each model
    for model in trained_models:
        model.latest_metric = model.metrics.first()
    
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save(commit=False)
            uploaded_file.user = request.user
            uploaded_file.original_name = request.FILES['file'].name
            uploaded_file.save()
            
            # Process the file to validate it
            try:
                process_uploaded_file(uploaded_file)
                messages.success(request, f'File "{uploaded_file.original_name}" uploaded successfully!')
                return redirect('dashboard')
            except Exception as e:
                uploaded_file.delete()  # Remove file if processing fails
                messages.error(request, f'Error processing file: {str(e)}')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = FileUploadForm()
    
    context = {
        'form': form,
        'files': files,
        'trained_models': trained_models,
    }
    return render(request, 'application/dashboard.html', context)


@login_required
def file_detail(request, pk):
    """Show details and preview of an uploaded file"""
    file = get_object_or_404(UploadedFile, pk=pk, user=request.user)
    
    # Get file preview data
    preview_data = None
    columns = []
    file_info = {}
    
    try:
        df = load_dataframe(file)
        preview_data = df.head(10).to_html(classes='table table-striped', index=False)
        columns = df.columns.tolist()
        file_info = {
            'rows': len(df),
            'columns': len(df.columns),
            'size': f"{file.file.size / 1024:.2f} KB",
            'missing_values': df.isnull().sum().sum(),
        }
    except Exception as e:
        messages.error(request, f'Error loading file preview: {str(e)}')
    
    context = {
        'file': file,
        'preview_data': preview_data,
        'columns': columns,
        'file_info': file_info,
    }
    return render(request, 'application/file_detail.html', context)


@login_required
def delete_file(request, pk):
    """Delete an uploaded file"""
    file = get_object_or_404(UploadedFile, pk=pk, user=request.user)
    
    if request.method == 'POST':
        file_name = file.original_name
        # Delete the physical file
        if file.file:
            if os.path.isfile(file.file.path):
                os.remove(file.file.path)
        file.delete()
        messages.success(request, f'File "{file_name}" deleted successfully!')
        return redirect('file_list')
    
    return render(request, 'dashboard/confirm_delete.html', {'file': file})

@login_required
def analyze_target(request, dataset_id, target_column):
    """Analyze target column and determine task type"""
    dataset = get_object_or_404(UploadedFile, pk=dataset_id, user=request.user)
    
    try:
        df = load_dataframe(dataset)
        validator = DataValidator()
        
        # Validate and analyze target
        target_info = validator.check_target_column(df, target_column)
        
        return JsonResponse({
            'success': True,
            'task_type': target_info['task_type'],
            'n_unique': target_info['n_unique'],
            'class_distribution': target_info.get('class_distribution'),
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

# Helper functions
def process_uploaded_file(uploaded_file):
    """
    Process and validate uploaded file
    Marks file as processed if successful
    """
    try:
        df = load_dataframe(uploaded_file)
        
        # Basic validation
        if df.empty:
            raise ValueError("The uploaded file is empty")
        
        if len(df.columns) == 0:
            raise ValueError("No columns found in the file")
        
        # Mark as processed
        uploaded_file.processed = True
        uploaded_file.save()
        
        return True
    except Exception as e:
        raise Exception(f"Failed to process file: {str(e)}")


def load_dataframe(uploaded_file):
    """Load file into pandas DataFrame based on file extension"""
    file_path = uploaded_file.file.path
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    elif ext == '.txt':
        # Try different delimiters for .txt files
        try:
            return pd.read_csv(file_path, sep='\t')
        except:
            try:
                return pd.read_csv(file_path, sep=',')
            except:
                return pd.read_csv(file_path, delim_whitespace=True)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


@login_required
def train_model(request, dataset_id):
    """Train a new model on a dataset"""
    dataset = get_object_or_404(UploadedFile, pk=dataset_id, user=request.user)
    
    if not dataset.processed:
        messages.error(request, 'Dataset is not yet processed. Please wait.')
        return redirect('application/file_detail', pk=dataset_id)
    
    # Get dataset columns for target selection
    try:
        df = load_dataframe(dataset)
        columns = df.columns.tolist()
    except Exception as e:
        messages.error(request, f'Error loading dataset: {str(e)}')
        return redirect('file_detail', pk=dataset_id)
    
    if request.method == 'POST':
        form = ModelTrainingForm(request.user, request.POST)
        if form.is_valid():
            # Create model record
            model = form.save(commit=False)
            model.user = request.user
            model.dataset = dataset
            model.framework = 'scikit-learn'
            model.status = 'training'
            
            # Store training parameters
            model.parameters = {
                'algorithm': form.cleaned_data['algorithm'],
                'test_size': form.cleaned_data['test_size'],
                'random_state': form.cleaned_data['random_state'],
            }
            model.save()
            
            # Start training (in production, use Celery for async)
            try:
                # Initialize trainer
                trainer = ModelTrainer(
                    dataset_path=dataset.file.path,
                    target_column=model.target_variable,
                    algorithm=form.cleaned_data['algorithm'],
                    test_size=form.cleaned_data['test_size'],
                    validation_size=form.cleaned_data.get('validation_size', 0.0),
                    random_state=form.cleaned_data['random_state'],
                    missing_value_strategy=form.cleaned_data.get('missing_value_strategy', 'mean'),
                )
                
                # Load and preprocess data
                trainer.load_and_preprocess_data()
                
                # Train model
                trainer.train_model()
                
                # Evaluate model
                metrics, cm, predictions = trainer.evaluate_model()
                
                # Save metrics
                PerformanceMetric.objects.create(
                    model=model,
                    accuracy=metrics['accuracy'],
                    precision=metrics['precision'],
                    recall=metrics['recall'],
                    f1_score=metrics['f1_score'],
                )
                
                # Save model file
                model_filename = f"model_{model.id}_{form.cleaned_data['algorithm']}.joblib"
                model_path = os.path.join(settings.MEDIA_ROOT, 'trained_models', model_filename)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                trainer.save_model(model_path)
                model.model_file = f'trained_models/{model_filename}'
                
                # Generate and save confusion matrix
                cm_plot = trainer.generate_confusion_matrix_plot()
                if cm_plot:
                    figure = Figure.objects.create(
                        model=model,
                        description='Confusion Matrix'
                    )
                    figure.figure_file.save(f'confusion_matrix_{model.id}.png', cm_plot)
                
                # Generate and save feature importance (if applicable)
                fi_plot = trainer.generate_feature_importance_plot()
                if fi_plot:
                    figure = Figure.objects.create(
                        model=model,
                        description='Feature Importance'
                    )
                    figure.figure_file.save(f'feature_importance_{model.id}.png', fi_plot)
                
                # Update model status
                model.status = 'completed'
                model.save()
                
                messages.success(request, f'Model "{model.name}" trained successfully with {metrics["accuracy"]:.2f}% accuracy!')
                return redirect('model_detail', pk=model.id)
                
            except Exception as e:
                model.status = 'failed'
                model.save()
                messages.error(request, f'Training failed: {str(e)}')
                return redirect('train_model', dataset_id=dataset_id)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = ModelTrainingForm(user=request.user, initial={'dataset': dataset})
    
    context = {
        'form': form,
        'dataset': dataset,
        'columns': columns,
    }
    return render(request, 'application/train_model.html', context)


@login_required
def model_detail(request, pk):
    """Display model details and metrics"""
    model = get_object_or_404(AIModel, pk=pk, user=request.user)
    metrics = model.metrics.first()
    figures = model.figures.all()
    
    context = {
        'model': model,
        'metrics': metrics,
        'figures': figures,
    }
    return render(request, 'application/model_detail.html', context)


@login_required
def model_list(request):
    """List all models for the current user"""
    models = AIModel.objects.filter(user=request.user).prefetch_related('metrics').order_by('-created_at')
    
    # Add latest metric to each model
    for model in models:
        model.latest_metric = model.metrics.first()
    
    context = {
        'models': models,
    }
    return render(request, 'application/model_list.html', context)

# Authentication Views
def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            # Add a user to database here
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    # Registration is a directory separate from the main application html templates
    return render(request, 'registration/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'registration/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')
