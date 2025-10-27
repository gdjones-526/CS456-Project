from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.conf import settings
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
    models = [
        "Linear Regression",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM",
        "Neural Network",
    ]
    context = {
        "models": models,
    }
    return render(request, "dashboard.html", context)

# @login_required is a django authentication library feature, 
# all views will use @login_required decorator