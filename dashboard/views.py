from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from core.models import UploadedFile, AIModel, PerformanceMetric, Figure
from .forms import FileUploadForm, ModelTrainingForm
from .ml_utils import *
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
import pandas as pd
import os
import json
from django.conf import settings
from django.http import JsonResponse
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.views.decorators.csrf import csrf_exempt
from collections import defaultdict # <-- Import this
from django.http import JsonResponse


def get_figure_url(request, model_id, figure_type):
    try:
        figure = Figure.objects.filter(model_id=model_id, description__icontains=figure_type).first()
        if figure and figure.figure_file:
            return JsonResponse({'url': figure.figure_file.url})
    except AIModel.DoesNotExist:
        pass
    return JsonResponse({'url': None})

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

@csrf_exempt
def analyze_selection(request, dataset_id):
    if request.method == "POST":
        try:
            body = json.loads(request.body)
            features = body.get("features", [])
            target = body.get("target")

            if not features or not target:
                return JsonResponse({"success": False, "error": "Missing features or target variable."})

            # Load dataset
            dataset = Dataset.objects.get(pk=dataset_id)
            df = pd.read_csv(dataset.file.path)

            if target not in df.columns or any(f not in df.columns for f in features):
                return JsonResponse({"success": False, "error": "Invalid column selection."})

            # Basic target type detection
            task_type = "classification" if df[target].nunique() < 10 else "regression"

            return JsonResponse({
                "success": True,
                "task_type": task_type,
                "n_features": len(features)
            })

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
        
@login_required
def dashboard(request):
    """Main dashboard view with file upload, management, and visualizations"""
    files = UploadedFile.objects.filter(user=request.user).order_by('-uploaded_at')
    trained_models = AIModel.objects.filter(user=request.user).prefetch_related('metrics', 'figures')
    
    grouped_models = defaultdict(lambda: {
        'classification': [],
        'regression': []
    })

    for model in trained_models:
        # Attach latest metric and figures
        model.latest_metric = model.metrics.first()
        model.visualizations = model.figures.all()
        
        dataset_name = model.dataset.original_name
        
        # Add model to the correct list
        if model.parameters['task_type'] == 'classification':
            grouped_models[dataset_name]['classification'].append(model)
        else:
            grouped_models[dataset_name]['regression'].append(model)

    sorted_grouped_models = {}
    for dataset_name, models in grouped_models.items():
        
        # Sort classification models by Accuracy (descending)
        # We handle 'None' values by treating them as -1 (lowest accuracy)
        sorted_class = sorted(
            models['classification'],
            key=lambda m: m.latest_metric.accuracy if m.latest_metric and m.latest_metric.accuracy is not None else -1,
            reverse=True  # Highest accuracy first
        )
        
        # Sort regression models by MAE (ascending)
        # We handle 'None' values by treating them as infinity (highest error)
        sorted_reg = sorted(
            models['regression'],
            key=lambda m: m.latest_metric.mae if m.latest_metric and m.latest_metric.mae is not None else float('inf'),
            reverse=False  # Lowest error first
        )
        
        if sorted_class or sorted_reg: # Only add datasets that have models
            sorted_grouped_models[dataset_name] = {
                'classification': sorted_class,
                'regression': sorted_reg
            }

    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save(commit=False)
            uploaded_file.user = request.user
            uploaded_file.original_name = request.FILES['file'].name
            uploaded_file.save()
            
            try:
                process_uploaded_file(uploaded_file)
                messages.success(request, f'File "{uploaded_file.original_name}" uploaded successfully!')
                return redirect('dashboard')
            except Exception as e:
                uploaded_file.delete()
                messages.error(request, f'Error processing file: {str(e)}')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = FileUploadForm()
    
    context = {
        'form': form,
        'files': files,
        'trained_models': trained_models,
        'grouped_models': sorted_grouped_models,
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
def train_model(request, dataset_id, algorithm=None):
    """Train a new model on a dataset"""
    dataset = get_object_or_404(UploadedFile, pk=dataset_id, user=request.user)
    if not dataset.processed:
        messages.error(request, 'Dataset is not yet processed. Please wait.')
        return redirect('application/file_detail', pk=dataset_id)
    
    # Get dataset columns for target selection and feature choices
    try:
        df = load_dataframe(dataset)
        columns = df.columns.tolist()
        feature_choices = [(col, col) for col in columns]

        # Determine sensible default features: prefer numeric columns, otherwise first up to 5 columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            default_features = numeric_cols
        else:
            default_features = columns[:5]

        # Initialize form with the dataset and choices
        initial_data = {'dataset': dataset}

        # For GET requests pre-select sensible defaults; for POST keep user's selection
        if request.method == 'POST':
            form = ModelTrainingForm(data=request.POST, initial=initial_data, user=request.user)
        else:
            initial_data['features'] = default_features
            form = ModelTrainingForm(initial=initial_data, user=request.user)

        # Set feature choices before processing the form and ensure initial values are shown
        form.fields['features'].choices = feature_choices
        # Only set the field's initial selection if not POST
        if request.method != 'POST':
            form.fields['features'].initial = default_features
            
    except Exception as e:
        messages.error(request, f'Error loading dataset: {str(e)}')
        return redirect('file_detail', pk=dataset_id)
    
    # ---------------
    classification_algorithms = {}
    regression_algorithms = {}
    
    # Build classification algorithms dict
    for key, model_info in ModelRegistry.CLASSIFICATION_MODELS.items():
        classification_algorithms[key] = {
            'name': model_info['name'],
            'description': model_info['description'],
            'family': model_info['family'],
            'task_type': 'classification'
        }
    
    # Build regression algorithms dict
    for key, model_info in ModelRegistry.REGRESSION_MODELS.items():
        regression_algorithms[key] = {
            'name': model_info['name'],
            'description': model_info['description'],
            'family': model_info['family'],
            'task_type': 'regression'
        }

    if request.method == 'POST':
        # Check validation and report errors before proceeding
        if not form.is_valid():
            for field, errors in form.errors.items():
                field_label = form.fields.get(field).label if form.fields.get(field) else field
                for error in errors:
                    messages.error(request, f"{field_label}: {error}")
        
        if form.is_valid():
            # Create model record
            model = form.save(commit=False)
            model.user = request.user
            model.dataset = dataset
            model.framework = 'scikit-learn'
            model.status = 'training'
            
            # Parse algorithm and task type from the selection
            # task_type, algorithm = form.cleaned_data['algorithm'].split('|')
            task_type = form.cleaned_data['task_type']
            algorithm = form.cleaned_data['algorithm']
            
            # Store training parameters
            model.parameters = {
                'algorithm': algorithm,
                'test_size': form.cleaned_data['test_size'],
                'random_state': form.cleaned_data['random_state'],
                'task_type': task_type,
            }
            model.save()
            
            try:
                test_size_val = form.cleaned_data.get('test_size', 0.2)
                random_state_val = form.cleaned_data.get('random_state', 42)

                # Get selected features from the form
                selected_features = form.cleaned_data.get('features', [])
                
                # Initialize trainer
                trainer = ModelTrainer(
                    dataset_path=dataset.file.path,
                    target_column=model.target_variable,
                    algorithm=algorithm,
                    test_size=test_size_val,
                    task_type=task_type,
                    validation_size=form.cleaned_data.get('validation_size', 0.0),
                    random_state=random_state_val,
                    missing_value_strategy=form.cleaned_data.get('missing_value_strategy', 'mean'),
                    selected_features=selected_features
                )
                
                # Load and preprocess data
                trainer.load_and_preprocess_data()
                
                # Train model
                trainer.train_model()
                
                # Evaluate model
                metrics, cm, predictions = trainer.evaluate_model()

                #  Save metrics - HANDLE BOTH CLASSIFICATION AND REGRESSION
                if form.cleaned_data['task_type'] == 'classification':
                    PerformanceMetric.objects.create(
                        model=model,
                        accuracy=metrics.get('accuracy'),
                        precision=metrics.get('precision'),
                        recall=metrics.get('recall'),
                        f1_score=metrics.get('f1_score'),
                        loss=metrics.get('loss'),
                    )
                else:  # regression
                    PerformanceMetric.objects.create(
                        model=model,
                        mse=metrics.get('mse'),
                        rmse=metrics.get('rmse'),
                        mae=metrics.get('mae'),
                        r2_score=metrics.get('r2_score'),
                        loss=metrics.get('mse'),
                        accuracy=None,
                        precision=None,
                        recall=None,
                        f1_score=None,
                    )

                # Save model file
                model_filename = f"model_{model.id}_{form.cleaned_data['algorithm']}.joblib"
                model_path = os.path.join(settings.MEDIA_ROOT, 'trained_models', model_filename)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                trainer.save_model(model_path)
                model.model_file = f'trained_models/{model_filename}'
                
                # Generate and save confusion matrix
                # cm_plot = trainer.generate_confusion_matrix_plot()
                # if cm_plot:
                #     figure = Figure.objects.create(
                #         model=model,
                #         description='Confusion Matrix'
                #     )
                #     figure.figure_file.save(f'confusion_matrix_{model.id}.png', cm_plot)

                # # Generate and save ROC curve (classification only)
                # roc_plot = trainer.generate_roc_curve_plot()
                # if roc_plot:
                #     figure = Figure.objects.create(
                #         model=model,
                #         description='ROC Curve'
                #     )
                #     figure.figure_file.save(f'roc_curve_{model.id}.png', roc_plot)

                
                # # Generate and save feature importance (if applicable)
                # fi_plot = trainer.generate_feature_importance_plot()
                # if fi_plot:
                #     figure = Figure.objects.create(
                #         model=model,
                #         description='Feature Importance'
                #     )
                #     figure.figure_file.save(f'feature_importance_{model.id}.png', fi_plot)
                
                # # Update model status
                # model.status = 'completed'
                # model.save()
                
                # --- Task-Specific Figure Generation ---
                # Get the task type from the trainer object
                task_type = trainer.task_type 

                if task_type == 'classification':
                    # Generate and save Confusion Matrix (classification only)
                    cm_plot = trainer.generate_confusion_matrix_plot()
                    if cm_plot:
                        figure = Figure.objects.create(
                            model=model,
                            description='Confusion Matrix'
                        )
                        figure.figure_file.save(f'confusion_matrix_{model.id}.png', cm_plot)

                    # Generate and save ROC curve (classification only)
                    roc_plot = trainer.generate_roc_curve_plot()
                    if roc_plot:
                        figure = Figure.objects.create(
                            model=model,
                            description='ROC Curve'
                        )
                        figure.figure_file.save(f'roc_curve_{model.id}.png', roc_plot)
                
                elif task_type == 'regression':
                    # Generate and save Actual vs. Predicted (regression only)
                    avp_plot = trainer.generate_actual_vs_predicted_plot()
                    if avp_plot:
                        figure = Figure.objects.create(
                            model=model,
                            description='Actual vs. Predicted'
                        )
                        figure.figure_file.save(f'actual_vs_predicted_{model.id}.png', avp_plot)

                    # Generate and save Residuals vs. Fitted (regression only)
                    rvf_plot = trainer.generate_residuals_vs_fitted_plot()
                    if rvf_plot:
                        figure = Figure.objects.create(
                            model=model,
                            description='Residuals vs. Fitted'
                        )
                        figure.figure_file.save(f'residuals_vs_fitted_{model.id}.png', rvf_plot)
                
                # Generate and save feature importance (if applicable)
                # This function works for both regression and classification
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
                
                messages.success(request, f'Model "{model.name}" trained successfully with ') #{metrics["accuracy"]:.2f}% accuracy!
                return redirect('model_detail', pk=model.id)
                
            except Exception as e:
                model.status = 'failed'
                model.save()
                messages.error(request, f'Training failed: {str(e)}')
                return redirect('train_model', dataset_id=dataset_id)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        pass
    
    context = {
        'form': form,
        'dataset': dataset,
        'columns': columns,
        'dataset': dataset,
        'classification_algorithms_json': json.dumps(classification_algorithms),
        'regression_algorithms_json': json.dumps(regression_algorithms),
    }
    return render(request, 'application/train_model.html', context)


@login_required
def model_detail(request, pk):
    """Display model details and metrics"""
    model = get_object_or_404(AIModel, pk=pk, user=request.user)
    metrics = model.metrics.first()
    figures = model.figures.all()
    
    # Load model file to get feature names and additional info
    feature_names = []
    preprocessing_info = {}
    
    if model.model_file:
        try:
            model_path = model.model_file.path
            model_data = joblib.load(model_path)
            
            # Extract feature names from saved model
            feature_names = model_data.get('feature_names', [])
            
            # Extract preprocessing configuration
            preprocessing_config = model_data.get('preprocessing_config', {})
            preprocessing_info = {
                'missing_value_strategy': preprocessing_config.get('missing_value_strategy', 'N/A'),
                'train_test_split': preprocessing_config.get('train_test_split', 0.2),
                'validation_split': preprocessing_config.get('validation_split', 0.0),
                'random_state': preprocessing_config.get('random_state', 42),
                'categorical_columns': preprocessing_config.get('categorical_columns', []),
                'numeric_columns': preprocessing_config.get('numeric_columns', []),
            }
            
        except Exception as e:
            print(f"Error loading model file: {e}")
            # Fallback: try to get from parameters if available
            if model.parameters and isinstance(model.parameters, dict):
                feature_names = model.parameters.get('feature_columns', [])
    
    # Get dataset info if available
    dataset_info = None
    if model.dataset:
        try:
            from .ml_utils import load_dataframe
            df = load_dataframe(model.dataset)
            dataset_info = {
                'rows': len(df),
                'columns': len(df.columns),
                'all_columns': df.columns.tolist(),
            }
        except Exception as e:
            print(f"Error loading dataset: {e}")
    
    context = {
        'model': model,
        'metrics': metrics,
        'figures': figures,
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'preprocessing_info': preprocessing_info,
        'dataset_info': dataset_info,
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

def delete_model(request, model_id):
    if request.method == 'POST':
        model = get_object_or_404(AIModel, id=model_id)

        if model.user != request.user:
            return JsonResponse({'error': 'Unauthorized'}, status=403)

        model.delete()
        return JsonResponse({'message': 'Model deleted successfully'}, status=200)

    return JsonResponse({'error': 'Invalid request'}, status=400)

# Authentication Views
def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
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
