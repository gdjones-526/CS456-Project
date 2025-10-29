from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.conf import settings
from django.contrib import messages
from core.models import UploadedFile
from .forms import *
import pandas as pd
import os

# from .models import models from models dir

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

# Main Application Views

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

# @login_required is a django authentication library feature, 
# all views will use @login_required decorator

@login_required
def upload_file(request):
    """Handle file upload and initial processing"""
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
                return redirect('file_list')
            except Exception as e:
                uploaded_file.delete()  # Remove file if processing fails
                messages.error(request, f'Error processing file: {str(e)}')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = FileUploadForm()
    
    return render(request, 'application/upload_file.html', {'form': form})


@login_required
def file_list(request):
    """Display all uploaded files for the current user"""
    files = UploadedFile.objects.filter(user=request.user).order_by('-uploaded_at')
    return render(request, 'dashboard/file_list.html', {'files': files})


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
    else:
        raise ValueError(f"Unsupported file format: {ext}")